import  torch
import  torch.nn.functional as F
from    torch import nn
import  numpy as np
from    sklearn.metrics import precision_recall_fscore_support
from    sklearn.metrics import roc_auc_score, accuracy_score

class BaseModel(nn.Module):
    def forward(self, x, edge_index=None, edge_type=None, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement " \
                        "forward to return node_emb, node_logits")

    def compute_node_loss(self, node_logits, y, mask=None):
        if mask is not None:
            node_logits = node_logits[mask]
            y = y[mask]
        return F.cross_entropy(node_logits, y)

    def score_edges(self, node_emb, edges, edge_types=None):
        raise NotImplementedError("Subclasses must implement score_edges")

    def compute_link_loss(self, node_emb, pos_edges, pos_types, 
                          neg_edges, neg_types, margin=1.0):
        
        pos_scores = self.score_edges(node_emb, pos_edges, pos_types)
        neg_scores = self.score_edges(node_emb, neg_edges, neg_types)

        out = torch.cat([pos_scores, neg_scores])
        ground_truth = torch.cat([torch.ones_like(pos_scores), 
                                  torch.zeros_like(neg_scores)])
        perm = torch.randperm(out.size(0)) # Shuffle tensors       
        shuffled_out = out[perm]

        shuffled_ground_truth = ground_truth[perm]
        return F.binary_cross_entropy_with_logits(shuffled_out, 
                                                    shuffled_ground_truth)

        
    
def train_concurrent_prediction(model, data,
                   optimizer,
                   train_pos_edges, train_pos_types,
                   train_neg_edges, train_neg_types,
                   val_pos_edges, val_pos_types,
                   val_neg_edges, val_neg_types,
                   threshold = 0.5 ,
                   epochs=50):
    """
    Train a link prediction model with stratified positive and negative edges.

    Args:
        model: PyG link prediction GNN model with methods
        data: PyG Data object containing node features and edges
        train_pos_edges, val_pos_edges: Positive edges for training/validation
        train_pos_types, val_pos_types: Edge types for positive edges
        train_neg_edges, val_neg_edges: Negative edges for training/validation
        train_neg_types, val_neg_types: Edge types for negative edges (-1 for negatives)
        threshold : prediction threshold as in binary classificaiton
        epochs: Number of training epochs
    """

    for epoch in range(1, epochs + 1):

        model.train()
                
        node_emb, logits = model(data.x, 
                            data.edge_index, 
                            data.edge_type, 
                            data.num_nodes)
        
        # visit loss        
        loss_1 = model.compute_node_loss(logits, data.y, data.train_mask)    


        pos_scores = model.score_edges(node_emb, train_pos_edges, train_pos_types)
        neg_scores = model.score_edges(node_emb, train_neg_edges, train_pos_types)    

        # link loss
        loss_2  = model.compute_link_loss(node_emb, 
                          train_pos_edges, train_pos_types, 
                          train_neg_edges, train_neg_types)
        
        loss = loss_1 + loss_2
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        model.eval()
        with torch.no_grad():
            mask = data.val_mask
            
            node_emb, _ = model(data.x, 
                                data.edge_index, 
                                data.edge_type, 
                                data.num_nodes)
                        
            
            val_loss = model.compute_link_loss(
                node_emb,
                val_pos_edges, val_pos_types,
                val_neg_edges, val_neg_types
            )
                        
            pos_scores = model.score_edges(node_emb, val_pos_edges, val_pos_types)
            neg_scores = model.score_edges(node_emb, val_neg_edges, val_pos_types)    

            # Link Predicitons                    
            labels = torch.cat([torch.ones_like(pos_scores), 
                                torch.zeros_like(neg_scores)]).numpy()        
            scores =  torch.cat([ torch.sigmoid(pos_scores), 
                                 torch.sigmoid(neg_scores)]).numpy()            
            auc = roc_auc_score(labels, scores)            
            preds = (scores > threshold).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(labels, 
                                                                    preds, 
                                                                    average="binary")
            
            print(f"Epoch {epoch:03d} | "
                f"Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}  "
                f"\n\tLink Prediction | AUC: {auc:.4f} | P: {precision:.4f} | " 
                f"R: {recall:.4f} | F1: {f1:.4f}")
            

            # Node Predicitons      
            x = node_emb[mask]
            preds = model.classifier(x).argmax(dim=1)            
            labels = data.y[mask]          
            precision, recall, f1, _ = precision_recall_fscore_support(labels, 
                                    preds,average='weighted',zero_division=np.nan)
            auc = accuracy_score(labels, preds)
            print(f"\tNode Prediction | AUC: {auc:.4f} | "
                  f"P: {precision:.4f} | R: {recall:.4f} | F1: {f1:.4f}")

