import  torch
from    torch_geometric.nn import MessagePassing
import  torch.nn.functional as F
from    torch import nn
from    geoopt.manifolds import PoincareBall

from .base import BaseModel

class  HGCN(MessagePassing):
    def __init__(self, in_channels, out_channels, 
                 num_edge_types, 
                 c = 1.0,
                 edge_set = None):
        super(HGCN, self).__init__(aggr='add')  

        self.manifold = PoincareBall(c=c) 

        # linear layers
        self.lin_self_weight    = torch.nn.Parameter(torch.randn(out_channels, 
                                                                in_channels))
        self.lin_self_bias      = torch.nn.Parameter(torch.randn(out_channels))

        self.rel_set = edge_set

        # relation-specific parameters
        class RelLinear(torch.nn.Module):
            def __init__(self, in_channels, out_channels):
                super(RelLinear, self).__init__()
                self.register_parameter('weight', 
                    torch.nn.Parameter(torch.empty(out_channels, 
                                                   in_channels)))
                self.register_parameter('bias',
                     torch.nn.Parameter(torch.empty(out_channels)))
                torch.nn.init.xavier_uniform_(self.weight)
                torch.nn.init.zeros_(self.bias)
        self.lin_rel = torch.nn.ModuleList([
            RelLinear(in_channels, out_channels) 
            for _ in range(num_edge_types)
        ])

        # Initialize self parameters
        torch.nn.init.xavier_uniform_(self.lin_self_weight)
        torch.nn.init.zeros_(self.lin_self_bias)

    def hyperbolic_linear(self, x, weight, bias):

        x_tangent = self.manifold.logmap0(x)        
        out = torch.matmul(x_tangent, weight.t()) + bias.unsqueeze(0)
        out = self.manifold.expmap0(out)

        return out

    def forward(self, x, edge_index, edge_type):

        x = self.manifold.expmap0(x) 

        # self-messages
        x_self = self.hyperbolic_linear(x, 
                                        self.lin_self_weight, 
                                        self.lin_self_bias)  

        # compute neighbor messages        
        aggr_out = self.propagate(edge_index, x=x, edge_type=edge_type)

        # combine self- and neighbor messages
        out = self.manifold.mobius_add(x_self, aggr_out)
        out = self.manifold.logmap0(out) 
        
        return out

    def message(self, x_j, edge_type):
        
        # initialize in Poincaré ball
        messages = torch.zeros(x_j.size(0), self.lin_rel[0].weight.size(0), 
                               device=x_j.device)    
        messages = self.manifold.expmap0(messages) 

        # compute relation-wise messages
        for r in range(len(self.lin_rel)):
            if self.rel_set is None:

                mask = (edge_type == r)
                messages[mask] = self.hyperbolic_linear(x_j[mask], 
                                        self.lin_rel[r].weight, 
                                        self.lin_rel[r].bias)
            else:    
        
                if r in self.rel_set:

                    mask = (edge_type == r)
                    messages[mask] = self.hyperbolic_linear(x_j[mask], 
                                        self.lin_rel[r].weight, 
                                        self.lin_rel[r].bias)

        return messages

    def update(self, aggr_out):   
        return aggr_out 
        return F.sigmoid(aggr_out)     



class MultiLayerHGNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, 
                 num_edge_types, 
                 num_layers = 1,
                 c = 1.0 ,
                 edge_set   =   None):
        super(MultiLayerHGNN, self).__init__()

        if c < 0 or c > 1:
            raise  ValueError(f"unknown conformity parameter {c}")   


        self.num_layers = num_layers
        self.layers     = torch.nn.ModuleList()
        
        self.edge_set   = edge_set
        
        current_channels = in_channels
        for _ in range(num_layers): 
        
            self.layers.append(HGCN(current_channels, 
                                                out_channels, 
                                                num_edge_types,
                                                edge_set = edge_set))
            
            current_channels = out_channels

    def forward(self, x, edge_index, edge_type):
        # propogate and compute embeddings through layers

        for _,layer in enumerate(self.layers):            
            x = layer(x, edge_index, edge_type)

        return x


class HNMP(BaseModel):
    def __init__(self, in_channels, out_channels, 
                 num_edge_types, num_node_types, 
                 patient_relations, ontology_relations,                 
                 relation_dict, 
                 kappa_1    = 0, 
                 kappa_2    = 1,
                 num_layers = 1):        
        super(HNMP, self).__init__()

        self.encoder_p = MultiLayerHGNN(in_channels, out_channels, 
                                num_edge_types, 
                                num_layers, 
                                c  = kappa_1,
                                edge_set = [relation_dict[e] 
                                           for e in patient_relations])
        
        self.encoder_q = MultiLayerHGNN(in_channels, out_channels, 
                                num_edge_types, 
                                num_layers, c  = kappa_2 ,
                                edge_set = [relation_dict[e] 
                                           for e in ontology_relations])
                
        self.num_edge_types = num_edge_types        
        self.rel_embeddings = torch.nn.Parameter(torch.randn(num_edge_types, 
                                                             out_channels))
        torch.nn.init.xavier_uniform_(self.rel_embeddings)

        self.classifier     = torch.nn.Linear(out_channels, num_node_types)

    def forward(self, x, edge_index, edge_type, num_nodes=None):
        # generate node embeddings

        node_emb = self.encoder_p(x, edge_index, edge_type) # on p edges
        node_emb = self.encoder_q(node_emb, edge_index, edge_type) # on q edges

        node_logits = self.classifier(node_emb)
        
        return node_emb, node_logits

    def score_edges(self, node_emb, edges, edge_types):
        if edges.dim() == 2 and edges.size(0) == 2:
            src, dst = edges
        elif edges.dim() == 2 and edges.size(1) == 2:
            src, dst = edges[:,0], edges[:,1]
        else:
            raise ValueError(f"Unsupported edge shape: {edges.shape}")
        
        rel_emb = self.rel_embeddings[edge_types]
        scores  = (node_emb[src] * rel_emb * node_emb[dst]).sum(dim=1)

        return scores



if __name__ == "__main__":
    """
from models.hnmp_rgcn import *


if dataset == 'eicu':
    patient_relations = ['http://ehrtoolkit.org/ontology/eICU/has_icd_diagnosis',
                    'http://ehrtoolkit.org/ontology/eICU/has_lab_event',
                    'http://ehrtoolkit.org/ontology/eICU/has_medication'
    ]
else:

    patient_relations = [  'https://biomedit.ch/rdf/ehr-toolkit/has_diagnosis_code',
    'https://biomedit.ch/rdf/ehr-toolkit/has_lab_code',
    'https://biomedit.ch/rdf/ehr-toolkit/has_medication_code']


ontology_relations = [ 'http://www.w3.org/2002/07/owl#sameAs' ,
    'http://www.w3.org/2000/01/rdf-schema#subClassOf'
]

model = HNMP(in_channels, out_channels, 
                num_edge_types = int(data.edge_type.max())+1, 
                num_node_types = int(data.y.max())+1, 
                patient_relations = patient_relations, 
                ontology_relations = ontology_relations,                 
                relation_dict = edge_dict, 
                kappa_1    = 1, 
                kappa_2    = 1,
                num_layers = 1)

optimizer = geoopt.optim.RiemannianAdam( model.parameters(),lr=0.01)


                 """