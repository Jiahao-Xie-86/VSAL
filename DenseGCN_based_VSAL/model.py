import torch
import torch.nn as nn
import torchvision.models as models

from torch_geometric.nn import DenseGCNConv
import torch.nn.functional as F
from torch_geometric.nn import GraphNorm, InstanceNorm

from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse  # For converting dense adjacency to edge index
import timm



# class GraphNorm(nn.Module):
#     def __init__(self, hidden_dim, eps=1e-5):
#         super(GraphNorm, self).__init__()
#         self.eps = eps
#         self.gamma = nn.Parameter(torch.ones(hidden_dim))
#         self.beta = nn.Parameter(torch.zeros(hidden_dim))

#     def forward(self, node_features):
#         if node_features.size(0) <= 1:  # If there is only one or no node
#             # Skip normalization, just apply scaling and shifting
#             return self.gamma * node_features + self.beta

#         # Normal case: compute mean and std over all nodes
#         mean = node_features.mean(dim=0, keepdim=True)
#         std = node_features.std(dim=0, keepdim=True) + self.eps
#         normed_features = (node_features - mean) / std
#         normed_features = self.gamma * normed_features + self.beta
#         return normed_features



class ConditionalGraphGenerator(nn.Module):
    def __init__(self, hidden_dim=128, output_dim=2):
        super(ConditionalGraphGenerator, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Dense GCN layers to process the input layout (x, y coordinates) and dense adjacency matrix
        self.gcn1 = DenseGCNConv(2, hidden_dim)  # From (x, y) coordinates to hidden_dim
        # self.gn1 = GraphNorm(hidden_dim)    # Batch normalization layer
        self.gcn2 = DenseGCNConv(hidden_dim, hidden_dim)
        # self.gn2 = GraphNorm(hidden_dim)

        # Linear layer to process the noise vector
        self.fc_noise = nn.Linear(128, hidden_dim)

        # Output layer to predict the updated node coordinates
        self.fc_out = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, z, input_layout, adj_matrix, num_nodes):
        """
        Arguments:
        - z: Noise vector for the GAN (batch_size, latent_dim)
        - input_layout: Input layout (batch_size, num_nodes, 2), where 2 is (x, y) coordinates
        - adj_matrix: Adjacency matrix (batch_size, num_nodes, num_nodes)
        - num_nodes: Number of nodes in the graph

        Returns:
        - generated_layout: Generated node coordinates (batch_size, num_nodes, output_dim)
        """
        batch_size = input_layout.size(0)      
        generated_layouts = []

        # Find the maximum number of nodes in this batch
        max_num_nodes = input_layout.size(1)

        for i in range(batch_size):
            # Extract the layout and adjacency matrix for this batch item
            num_nodes_i = num_nodes[i].item()
            
            # Adjust the layout to only include the actual nodes
            layout = input_layout[i][:num_nodes_i, :]  # Shape: (num_nodes_i, 2)
            
            
            # Adjust the adjacency matrix to match the number of actual nodes
            adj = adj_matrix[i][:num_nodes_i, :num_nodes_i]  # Shape: (num_nodes_i, num_nodes_i)
        
            # Apply Dense GCN layers on the input layout (x, y coordinates) and dense adjacency matrix
            # node_features = torch.relu(self.gn1(self.gcn1(layout.unsqueeze(0), adj.unsqueeze(0)))) 
            # node_features = torch.relu(self.gn2(self.gcn2(node_features, adj.unsqueeze(0))))  # Shape: (1, num_nodes_i, hidden_dim) 

            node_features = torch.relu(self.gcn1(layout.unsqueeze(0), adj.unsqueeze(0)))
            node_features = torch.relu(self.gcn2(node_features, adj.unsqueeze(0))) # Shape: (1, num_nodes_i, hidden_dim) 

            

            # Remove the batch dimension after DenseGCNConv processing
            node_features = node_features.squeeze(0)
        
            # Encode the noise vector and repeat it to match num_nodes_i
            z_encoding = torch.relu(self.fc_noise(z[i])).unsqueeze(0).repeat(num_nodes_i, 1)

            # Concatenate node features and noise encoding
            conditioned_input = torch.cat([node_features, z_encoding], dim=-1)  # Shape: (num_nodes_i, hidden_dim * 2)

            # Generate new node coordinates (layout)
            generated_layout = self.fc_out(conditioned_input)  # Shape: (num_nodes, output_dim)

            # Pad the generated layout to the max number of nodes
            padded_layout = F.pad(generated_layout, (0, 0, 0, max_num_nodes - num_nodes_i), value=0)  # Shape: (max_num_nodes, output_dim)

            
            generated_layouts.append(padded_layout)

        # Stack all generated layouts into a single tensor
        generated_layouts = torch.stack(generated_layouts)

        return generated_layouts


# class ConditionalGraphGenerator(nn.Module):
#     def __init__(self, hidden_dim=128, output_dim=2, heads=4, num_layers=2):
#         super(ConditionalGraphGenerator, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.output_dim = output_dim
#         self.heads = heads
#         self.num_layers = num_layers

#         # First GAT layer: From (x, y) coordinates to hidden_dim * heads
#         self.gat1 = GATConv(2, hidden_dim, heads=heads, concat=True)
#         self.gn1 = GraphNorm(hidden_dim * heads)

#         # Additional GAT layers
#         self.gats = nn.ModuleList([GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True) for _ in range(num_layers - 1)])
#         self.gns = nn.ModuleList([GraphNorm(hidden_dim * heads) for _ in range(num_layers - 1)])

#         # Linear layer for noise processing
#         self.fc_noise = nn.Linear(128, hidden_dim * heads)

#         # Output layer to predict the updated node coordinates
#         self.fc_out = nn.Linear(hidden_dim * 2 * heads, output_dim)

#     def forward(self, z, input_layout, adj_matrix, num_nodes):
#         batch_size = input_layout.size(0)
#         generated_layouts = []
#         max_num_nodes = input_layout.size(1)

#         for i in range(batch_size):
#             num_nodes_i = num_nodes[i].item()
#             layout = input_layout[i][:num_nodes_i, :]  # (num_nodes_i, 2)
#             adj = adj_matrix[i][:num_nodes_i, :num_nodes_i]  # (num_nodes_i, num_nodes_i)

#             # Convert dense adjacency matrix to sparse edge index
#             edge_index, _ = dense_to_sparse(adj)

#             # Apply first GAT layer
#             node_features = torch.relu(self.gn1(self.gat1(layout, edge_index)))

#             # Apply additional GAT layers
#             for gat, gn in zip(self.gats, self.gns):
#                 node_features = torch.relu(gn(gat(node_features, edge_index)))

#             # Encode the noise vector and repeat it to match num_nodes_i
#             z_encoding = torch.relu(self.fc_noise(z[i])).unsqueeze(0).repeat(num_nodes_i, 1)

#             # Concatenate node features and noise encoding
#             conditioned_input = torch.cat([node_features, z_encoding], dim=-1)

#             # Generate new node coordinates
#             generated_layout = self.fc_out(conditioned_input)

#             # Pad the generated layout to the max number of nodes
#             padded_layout = F.pad(generated_layout, (0, 0, 0, max_num_nodes - num_nodes_i), value=0)
#             generated_layouts.append(padded_layout)

#         # Stack all generated layouts into a single tensor
#         return torch.stack(generated_layouts)


# ---- Discriminator for GAN ----

class GraphDiscriminator(nn.Module):
    def __init__(self, hidden_dim=128):
        super(GraphDiscriminator, self).__init__()
        self.hidden_dim = hidden_dim

        # Dense GCN layers to process the node coordinates and dense adjacency matrix
        self.gcn1 = DenseGCNConv(2, hidden_dim)
        # self.gn1 = GraphNorm(hidden_dim)   # Batch normalization layer
        self.gcn2 = DenseGCNConv(hidden_dim, hidden_dim)
        # self.gn2 = GraphNorm(hidden_dim)

        # Final linear layer for binary classification (real/fake)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, input_layout, adj_matrix):
        """
        Arguments:
        - input_layout: Node coordinates (batch_size, num_nodes, 2)
        - adj_matrix: Adjacency matrix (batch_size, num_nodes, num_nodes)

        Returns:
        - validity: Real/fake score (batch_size, 1)
        """
        batch_size = input_layout.size(0)
        validity_scores = []
        for i in range(batch_size):
            # Extract the layout and adjacency matrix for this batch item
            layout = input_layout[i]  # Shape: (num_nodes, 2)
            adj = adj_matrix[i]       # Shape: (num_nodes, num_nodes)

            # Apply Dense GCN layers on the input layout (x, y coordinates) and dense adjacency matrix
            # node_features = torch.relu(self.gn1(self.gcn1(layout.unsqueeze(0), adj.unsqueeze(0))))
            # node_features = torch.relu(self.gn2(self.gcn2(node_features, adj.unsqueeze(0))))  # Shape: (1, num_nodes, hidden_dim)

            node_features = torch.relu(self.gcn1(layout.unsqueeze(0), adj.unsqueeze(0)))
            node_features = torch.relu(self.gcn2(node_features, adj.unsqueeze(0)))  # Shape: (1, num_nodes, hidden_dim)

            # Remove the batch dimension after DenseGCNConv processing
            node_features = node_features.squeeze(0)

            # Global pooling (mean over nodes)
            pooled_features = torch.mean(node_features, dim=0)  # Shape: (hidden_dim,)

            # Final binary classification (real/fake)
            validity = self.fc_out(pooled_features)  # Shape: (1,)
            validity_scores.append(validity)

        # Stack all validity scores into a single tensor
        validity_scores = torch.stack(validity_scores).squeeze(1)  # Shape: (batch_size, 1)

        return validity_scores

# class GraphDiscriminator(nn.Module):
#     def __init__(self, hidden_dim=128, heads=4, num_layers=2):
#         super(GraphDiscriminator, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.heads = heads
#         self.num_layers = num_layers

#         # First GAT layer: From (x, y) coordinates to hidden_dim * heads
#         self.gat1 = GATConv(2, hidden_dim, heads=heads, concat=True)
#         self.gn1 = GraphNorm(hidden_dim * heads)

#         # Additional GAT layers
#         self.gats = nn.ModuleList([GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True) for _ in range(num_layers - 1)])
#         self.gns = nn.ModuleList([GraphNorm(hidden_dim * heads) for _ in range(num_layers - 1)])

#         # Final linear layer for binary classification (real/fake)
#         self.fc_out = nn.Linear(hidden_dim * heads, 1)

#     def forward(self, input_layout, adj_matrix):
#         batch_size = input_layout.size(0)
#         validity_scores = []

#         for i in range(batch_size):
#             num_nodes_i = input_layout[i].size(0)
#             layout = input_layout[i]  # Shape: (num_nodes_i, 2)
#             adj = adj_matrix[i][:num_nodes_i, :num_nodes_i]  # Shape: (num_nodes_i, num_nodes_i)

#             # Convert dense adjacency matrix to sparse edge index
#             edge_index, _ = dense_to_sparse(adj)

#             # Apply first GAT layer
#             node_features = torch.relu(self.gn1(self.gat1(layout, edge_index)))

#             # Apply additional GAT layers
#             for gat, gn in zip(self.gats, self.gns):
#                 node_features = torch.relu(gn(gat(node_features, edge_index)))

#             # Global pooling (mean over nodes)
#             pooled_features = torch.mean(node_features, dim=0)  # Shape: (hidden_dim,)

#             # Final binary classification (real/fake)
#             validity = self.fc_out(pooled_features)
#             validity_scores.append(validity)

#         # Stack all validity scores into a single tensor
#         return torch.stack(validity_scores).squeeze(1)



def get_resnet50_classifier():
    resnet50 = models.resnet50(pretrained=True)
    # Modify the final layer to classify Hamiltonian cycle (binary classification)
    num_ftrs = resnet50.fc.in_features
    resnet50.fc = nn.Linear(num_ftrs, 2)
    #return resnet50.to(device)  # Move the classifier to the device
    return resnet50

def get_efficientnet_classifier():
    # Load ImageNet-pretrained EfficientNet models

    # load B0 224×224
    model = timm.create_model('efficientnet_b0', pretrained=True)

    # load B4 380×380
    # model = timm.create_model('efficientnet_b4', pretrained=True)

    # Replace classifier head (nn.Linear) for binary classification
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 2)

    # #load B6 528×528
    # weights = EfficientNet_B6_Weights.IMAGENET1K_V1

    # model = efficientnet_b6(weights=weights)

    # # Replace classifier head (nn.Linear) for binary classification
    # in_features = model.classifier[1].in_features
    # model.classifier[1] = nn.Linear(in_features, 2)

    return model

def get_vit_classifier():
    """
    Returns a Vision Transformer (ViT) model for binary classification.
    """
    # Load a pre-trained ViT model from the timm library
    vit = timm.create_model('vit_base_patch16_224', pretrained=True)
    
    # Get the number of input features for the classifier head
    num_ftrs = vit.head.in_features
    
    # Modify the head to classify into 2 classes (Hamiltonian vs Non-Hamiltonian)
    vit.head = nn.Linear(num_ftrs, 2)
    
    return vit
