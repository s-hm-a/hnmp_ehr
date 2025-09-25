import  torch
import  torch.nn.functional as F
from    torch import nn

class BaseModel(nn.Module):
    def forward(self, x, edge_index=None, edge_type=None, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward to return node_emb, node_logits")

    def compute_node_loss(self, node_logits, y, mask=None):
        if mask is not None:
            node_logits = node_logits[mask]
            y = y[mask]
        return F.cross_entropy(node_logits, y)

    def score_edges(self, node_emb, edges, edge_types=None):
        raise NotImplementedError("Subclasses must implement score_edges")

    def compute_link_loss(self, node_emb, pos_edges, pos_types, neg_edges, neg_types, margin=1.0):
        
        pos_scores = self.score_edges(node_emb, pos_edges, pos_types)
        neg_scores = self.score_edges(node_emb, neg_edges, neg_types)

        num_pairs = min(pos_scores.size(0), neg_scores.size(0))
        
        pos_scores = pos_scores[:num_pairs]
        neg_scores = neg_scores[:num_pairs]
        target = torch.ones(num_pairs)

        return F.margin_ranking_loss(pos_scores, neg_scores, target, margin=margin)
