import torch
import torch.nn as nn
import torch.nn.functional as F
from geoopt.manifolds import PoincareBall

from base import BaseModel,train_concurrent_prediction

MIN_NORM = 1e-15
BALL_EPS = {torch.float32: 4e-3, torch.float64: 1e-5}

def artanh(x):
    return x.clamp(-1 + 1e-5, 1 - 1e-5).atanh()


def tanh(x):
    return x.clamp(-15, 15).tanh()


def expmap0(u, c):
    sqrt_c = c ** 0.5
    u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    return project(gamma_1, c)


def logmap0(y, c):
    sqrt_c = c ** 0.5
    y_norm = y.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    return y / y_norm / sqrt_c * artanh(sqrt_c * y_norm)


def project(x, c):
    norm = x.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    eps = BALL_EPS[x.dtype]
    maxnorm = (1 - eps) / (c ** 0.5)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)


def mobius_add(x, y, c):
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    xy = torch.sum(x * y, dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return num / denom.clamp_min(MIN_NORM)


class HyperbolicGIE(BaseModel):
    def __init__(self, num_entities, num_relations, num_classes,dim=64, c=0.2):
        super().__init__()
        self.manifold = PoincareBall(c=c)
        self.c = c
        self.dim = dim

        # Entity & relation embeddings
        self.entity = nn.Embedding(num_entities, dim)
        self.rel = nn.Embedding(num_relations, 2 * dim)
        self.context_vec = nn.Embedding(num_relations, dim)

        nn.init.normal_(self.entity.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.rel.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.context_vec.weight, mean=0.0, std=0.01)

        # Relation rotation embeddings
        self.rel_diag1 = nn.Embedding(num_relations, dim)
        self.rel_diag2 = nn.Embedding(num_relations, dim)
        nn.init.uniform_(self.rel_diag1.weight, -1.0, 1.0)
        nn.init.uniform_(self.rel_diag2.weight, -1.0, 1.0)

        # Learnable curvature per relation
        self.c1 = nn.Parameter(torch.ones(num_relations, 1))
        self.c2 = nn.Parameter(torch.ones(num_relations, 1))
        self.c_rel = nn.Parameter(torch.ones(num_relations, 1))

        self.num_classes = num_classes  # set this when initializing the model
        self.classifier = nn.Linear(dim, num_classes)

        self.act = nn.Softmax(dim=1)
        self.scale = 1.0 / (dim ** 0.5)

    def mobius_linear(self, x, weight):
        """Mobius linear transform in hyperbolic space."""
        x_tan = self.manifold.logmap0(x)
        out = torch.matmul(x_tan, weight.t())
        return self.manifold.expmap0(out)

    def get_queries(self, queries):
        """
        queries: [batch, 3] (head, relation, tail)
        Returns: hyperbolic embeddings for head+relation
        """
        heads = queries[:, 0]
        rels = queries[:, 1]

        # Curvatures
        batch_c1 = F.softplus(self.c1[rels])
        batch_c2 = F.softplus(self.c2[rels])
        batch_c = F.softplus(self.c_rel[rels])

        # Head embeddings in two curvatures
        head1 = expmap0(self.entity(heads), batch_c1)
        head2 = expmap0(self.entity(heads), batch_c2)

        # Relation split
        rel_emb = self.rel(rels)
        rel1, rel2 = torch.chunk(rel_emb, 2, dim=1)
        rel1 = expmap0(rel1, batch_c1)
        rel2 = expmap0(rel2, batch_c2)

        # Mobius addition
        lhs1 = project(mobius_add(head1, rel1, batch_c1), batch_c1)
        lhs2 = project(mobius_add(head2, rel2, batch_c2), batch_c2)

        # Apply rotations (simple element-wise)
        lhs1 = lhs1 * self.rel_diag1(rels)
        lhs2 = lhs2 * self.rel_diag2(rels)

        # Reshape to [B,1,dim] for attention
        lhs1 = lhs1.unsqueeze(1)
        lhs2 = lhs2.unsqueeze(1)
        head_exp = self.entity(heads).unsqueeze(1)

        # Attention across embeddings
        cands = torch.cat([lhs1, lhs2, head_exp], dim=1)  # [B,3,dim]
        context = self.context_vec(rels).unsqueeze(1)  # [B,1,dim]
        att_weights = torch.sum(context * cands * self.scale, 
                                dim=-1, keepdim=True)  # [B,3,1]
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)  # [B, dim]

        # Final Mobius add with relation
        rel_final = expmap0(rel1, batch_c)
        lhs_final = expmap0(att_q, batch_c)
        res = project(mobius_add(lhs_final, rel_final, batch_c), batch_c)
        return res, batch_c

    def forward(self, x=None, edge_index=None, edge_type=None, num_nodes=None):

        all_nodes = torch.arange(self.entity.num_embeddings)
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
        return -sqdist


if __name__ == '__main__':


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

    model = HyperbolicGIE(num_entities = data.num_nodes, 
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


