
import torch
from torch import nn

from .base import BaseModel
from torch_geometric.nn import MetaPath2Vec

class Metapath2VecClassifier(BaseModel):
    def __init__(self, edge_index, edge_type,num_node_types=2, num_nodes=769, embedding_dim=64, 
                 walk_length=20, context_size=5, walks_per_node=10, num_negative_samples=5):
        super().__init__()
        self.hidden_dim = embedding_dim + 16  # Concatenate with data.x [769, 16]
        self.num_nodes = num_nodes
        
        # Define metapath: node -> edge_type -> node
        unique_edge_types = torch.unique(edge_type).cpu().numpy()
        self.num_edge_types = len(unique_edge_types)
        metapath = [('node', f'type_{et}', 'node') for et in unique_edge_types]
        
        # Create edge_index_dict
        edge_index_dict = {('node', f'type_{et}', 'node'): edge_index[:, edge_type == et] 
                           for et in unique_edge_types}
        
        # Initialize MetaPath2Vec
        self.model = MetaPath2Vec(
            edge_index_dict=edge_index_dict,
            embedding_dim=embedding_dim,
            metapath=metapath,
            walk_length=walk_length,
            context_size=context_size,
            walks_per_node=walks_per_node,
            num_negative_samples=num_negative_samples,
            sparse=False
        )
        
        # Linear layer to combine data.x and embeddings
        self.combine = nn.Linear(16 + embedding_dim, self.hidden_dim)

        # Node classifier
        self.classifier = nn.Linear(self.hidden_dim, num_node_types)  # for binary classification

        # Relation embeddings for link prediction
        self.rel_embeddings = nn.Parameter(torch.randn(self.num_edge_types, self.hidden_dim))
        nn.init.xavier_uniform_(self.rel_embeddings, gain=0.01)

    def forward(self, x, edge_index=None, edge_type=None, num_nodes=None):
        num_nodes = x.size(0) if num_nodes is None else num_nodes
        
        # Get MetaPath2Vec embeddings for all nodes
        embeddings = self.model('node', batch=torch.arange(num_nodes, device=x.device))
        
        # Concatenate features + embeddings
        combined = torch.cat([x, embeddings], dim=1)
        node_emb = self.combine(combined)
        
        # Node classification logits
        node_logits = self.classifier(node_emb)
        
        return node_emb, node_logits



    def score_edges(self, node_emb, edge_index_tensor, edge_types):
        src, dst = edge_index_tensor
        edge_types = edge_types.clamp(0, self.num_edge_types - 1)  # fix here
        rel_emb = self.rel_embeddings[edge_types]
        scores = torch.sum(node_emb[src] * node_emb[dst] * rel_emb, dim=-1)
        return scores

