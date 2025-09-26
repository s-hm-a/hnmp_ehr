
import torch
from torch import nn

from base import BaseModel,train_concurrent_prediction
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
        self.classifier = nn.Linear(self.hidden_dim, num_node_types) 

        # Relation embeddings for link prediction
        self.rel_embeddings = nn.Parameter(torch.randn(self.num_edge_types, 
                                                       self.hidden_dim))
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
        edge_types = edge_types.clamp(0, self.num_edge_types - 1) 
        rel_emb = self.rel_embeddings[edge_types]
        scores = torch.sum(node_emb[src] * node_emb[dst] * rel_emb, dim=-1)
        return scores



if __name__ == "__main__":


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

    embedding_dim         =  data.x.shape[1]

    model = Metapath2VecClassifier(   edge_index=data.edge_index,
        edge_type=data.edge_type,
        num_node_types=int(data.y.max().item() + 1),
        num_nodes=data.x.size(0),
                           embedding_dim=embedding_dim )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_concurrent_prediction(model, data,
                    optimizer,
                    train_pos_edges, train_pos_types,
                    train_neg_edges, train_neg_types,
                    val_pos_edges, val_pos_types,
                    val_neg_edges, val_neg_types,
                    epochs=epochs)
    


