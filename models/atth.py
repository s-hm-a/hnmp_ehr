import torch
import torch.nn as nn
import torch.nn.functional as F
from geoopt.manifolds import PoincareBall

from base import BaseModel, train_concurrent_prediction

# Utility functions for hyperbolic operations
MIN_NORM = 1e-15
BALL_EPS = {torch.float32: 4e-3, torch.float64: 1e-5}

def project(x, c):
    norm = x.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    eps = BALL_EPS[x.dtype]
    maxnorm = (1 - eps) / (c ** 0.5)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)

def tanh(x):
    return x.clamp(-15, 15).tanh()

def artanh(x):
    x = x.clamp(-1 + 1e-5, 1 - 1e-5)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))

def expmap0(u, c):
    sqrt_c = c ** 0.5
    u_norm = u.norm(dim=-1, keepdim=True).clamp_min(MIN_NORM)
    gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    return project(gamma_1, c)

def logmap0(y, c):
    sqrt_c = c ** 0.5
    y_norm = y.norm(dim=-1, keepdim=True).clamp_min(MIN_NORM)
    return y / y_norm / sqrt_c * artanh(sqrt_c * y_norm)

def mobius_add(x, y, c):
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    xy = torch.sum(x * y, dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c**2 * x2 * y2
    return num / denom.clamp_min(MIN_NORM)

# Attention-based Hyperbolic KG Embedding model
class AttH(BaseModel):
    def __init__(self, num_entities, num_relations,num_classes, dim, c=0.2):
        super().__init__()
        self.dim = dim
        self.c = c
        self.manifold = PoincareBall(c=c)

        # Entity embeddings
        self.entity = nn.Embedding(num_entities, dim)
        nn.init.normal_(self.entity.weight, mean=0.0, std=0.01)

        # Relation embeddings (for translation)
        self.rel = nn.Embedding(num_relations, dim)
        nn.init.normal_(self.rel.weight, mean=0.0, std=0.01)

        # Context vectors and rotation parameters
        self.context_vec = nn.Embedding(num_relations, dim)
        self.rel_diag = nn.Embedding(num_relations, dim)
        nn.init.normal_(self.context_vec.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.rel_diag.weight, mean=0.0, std=0.01)

        # Scaling for attention
        self.scale = 1.0 / (dim ** 0.5)
        self.act = nn.Softmax(dim=1)

        self.classifier = nn.Linear(dim, num_classes)

        # Curvature
        self.c_param = nn.Parameter(torch.ones(1) * c, requires_grad=True)

    def get_queries(self, queries):
        """
        queries: [B, 3] with (head_idx, rel_idx, tail_idx)
        Returns node embeddings in hyperbolic space
        """
        heads = self.entity(queries[:, 0])
        rels = self.rel(queries[:, 1])
        c = F.softplus(self.c_param)

        # Hyperbolic translation
        head_h = expmap0(heads, c)
        rel_h = expmap0(rels, c)
        lhs = mobius_add(head_h, rel_h, c)

        # Attention over multiple embeddings
        diag = self.rel_diag(queries[:, 1])
        cands = torch.stack([lhs, diag], dim=1)  # [B, 2, dim]
        context = self.context_vec(queries[:, 1]).unsqueeze(1)
        att_weights = torch.sum(context * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_output = torch.sum(att_weights * cands, dim=1)
        out = expmap0(att_output, c)

        return out, None

    def forward(self, x=None, edge_index=None, edge_type=None, num_nodes=None):
        """
        Compatible with train_linkpred:
        returns embeddings for all nodes
        """
        if num_nodes is None:
            num_nodes = self.entity.num_embeddings
        all_nodes = torch.arange(num_nodes, device=self.entity.weight.device)
        # queries: relation index 0 is placeholder
        queries = torch.stack([all_nodes, torch.zeros_like(all_nodes), 
                               all_nodes], dim=1)
        node_emb, _ = self.get_queries(queries)

        logits = self.classifier(node_emb)

        return node_emb, logits

    def score_edges(self, node_emb, edge_index, edge_type):
        u, v = edge_index
        u_e = node_emb[u]
        v_e = node_emb[v]
        c = self.c
        sqdist = (2 * torch.atanh(
            torch.clamp(torch.norm(mobius_add(-u_e, v_e, c), dim=-1), 
                        1e-10, 1 - 1e-5)
        )) ** 2
        return  -sqdist


if __name__ == "__main__":


    import geoopt
    import pickle
    import argparse

    parser = argparse.ArgumentParser(description='Script for Ontology-aligned model training.')
    parser.add_argument('--datafile', type=str, default='mimic4_data.pk', help='Dataset file name')
    parser.add_argument('--data_path', type=str, default='../data/', help='Path to data directory')
    parser.add_argument('--learning_rate', type=float, default=1e-2, help='Learning rate for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    args = parser.parse_args()

    dataset = args.datafile 
    data_path = args.data_path
    data_filename = data_path+dataset
    learning_rate = args.learning_rate
    epochs = args.epochs

    with open(data_filename, "rb") as f:
        loaded_data = pickle.load(f)

    edge_dict           = loaded_data["edge_dict"]
    data                = loaded_data["data"]
    train_pos_edges     = loaded_data['train_pos_edges']
    val_pos_edges       = loaded_data['val_pos_edges']
    train_pos_types     = loaded_data['train_pos_types']
    val_pos_types       = loaded_data['val_pos_types']
    train_neg_edges     = loaded_data['train_neg_edges']
    val_neg_edges       = loaded_data['val_neg_edges']
    train_neg_types     = loaded_data['train_neg_types']
    val_neg_types       = loaded_data['val_neg_types']

    in_channels         =  data.x.shape[1]

    model = AttH(num_entities = data.num_nodes, 
                num_relations = len(edge_dict),
                num_classes = int(data.y.max())+1, 
                dim = in_channels)

    optimizer = geoopt.optim.RiemannianAdam( model.parameters(),lr=0.01)

    train_concurrent_prediction(model, data,
                    optimizer,
                    train_pos_edges, train_pos_types,
                    train_neg_edges, train_neg_types,
                    val_pos_edges, val_pos_types,
                    val_neg_edges, val_neg_types,
                    epochs=epochs)