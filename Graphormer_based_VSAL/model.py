import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import networkx as nx
import numpy as np

from torch_geometric.nn import DenseGCNConv
from torch_geometric.nn import GraphNorm, InstanceNorm
from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse
import timm
from torchvision.models import efficientnet_b6, EfficientNet_B6_Weights

# Graphormer components
class GraphormerEncoder(nn.Module):
    """
    Graphormer encoder that can handle variable graph sizes and float tensors.
    This is a simplified version that captures the key Graphormer concepts.
    """
    def __init__(self, input_dim, hidden_dim, num_heads=8, num_layers=2, max_dist=20, dropout=0.1):
        super(GraphormerEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_dist = max_dist
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Spatial encoding (learnable distance embeddings)
        self.spatial_encoding = nn.Embedding(max_dist + 1, hidden_dim)
        
        # Degree encoding
        self.degree_encoding = nn.Embedding(1000, hidden_dim)  # Support up to 1000 nodes
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # Feed-forward networks
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms1 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.layer_norms2 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        
        # Final projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def compute_graph_features(self, adj_matrix):
        """
        Compute graph structural features for Graphormer.
        
        Args:
            adj_matrix: (batch_size, num_nodes, num_nodes) adjacency matrix
            
        Returns:
            spatial_pos: (batch_size, num_nodes, num_nodes) spatial positions
            in_degree: (batch_size, num_nodes) in-degree of each node
            out_degree: (batch_size, num_nodes) out-degree of each node
        """
        batch_size, num_nodes, _ = adj_matrix.shape
        device = adj_matrix.device
        
        # Initialize tensors
        spatial_pos = torch.full((batch_size, num_nodes, num_nodes), self.max_dist, 
                               dtype=torch.long, device=device)
        in_degree = torch.zeros(batch_size, num_nodes, dtype=torch.long, device=device)
        out_degree = torch.zeros(batch_size, num_nodes, dtype=torch.long, device=device)
        
        for b in range(batch_size):
            # Convert to numpy for networkx operations
            adj_np = adj_matrix[b].detach().cpu().numpy()
            G = nx.from_numpy_array(adj_np)
            
            # Compute shortest path distances
            try:
                lengths = dict(nx.all_pairs_shortest_path_length(G, cutoff=self.max_dist))
                for i in range(num_nodes):
                    for j in range(num_nodes):
                        if j in lengths.get(i, {}):
                            spatial_pos[b, i, j] = min(lengths[i][j], self.max_dist)
                        else:
                            spatial_pos[b, i, j] = self.max_dist
            except:
                # If graph is disconnected, use max_dist
                pass
            
            # Compute degrees
            degrees = dict(G.degree())
            for i in range(num_nodes):
                if i in degrees:
                    out_degree[b, i] = min(degrees[i], 999)  # Cap at 999 for embedding
                    in_degree[b, i] = min(degrees[i], 999)
        
        return spatial_pos, in_degree, out_degree
    
    def forward(self, node_features, adj_matrix):
        """
        Forward pass of Graphormer encoder.
        
        Args:
            node_features: (batch_size, num_nodes, input_dim) node features
            adj_matrix: (batch_size, num_nodes, num_nodes) adjacency matrix
            
        Returns:
            output: (batch_size, num_nodes, hidden_dim) encoded node features
        """
        batch_size, num_nodes, _ = node_features.shape
        
        # Compute graph structural features
        spatial_pos, in_degree, out_degree = self.compute_graph_features(adj_matrix)
        
        # Input projection
        x = self.input_proj(node_features)  # (batch_size, num_nodes, hidden_dim)
        
        # Add structural encodings
        spatial_enc = self.spatial_encoding(spatial_pos.view(-1)).view(batch_size, num_nodes, num_nodes, -1)
        degree_enc = self.degree_encoding(in_degree) + self.degree_encoding(out_degree)
        
        # Add degree encoding to node features
        x = x + degree_enc
        
        # Apply transformer layers
        for i in range(self.num_layers):
            # Self-attention with spatial bias
            attn_input = x + spatial_enc.mean(dim=2)  # Average spatial encoding across nodes
            
            # Multi-head attention
            attn_output, _ = self.attention_layers[i](attn_input, attn_input, attn_input)
            attn_output = self.layer_norms1[i](x + attn_output)
            
            # Feed-forward network
            ffn_output = self.ffn_layers[i](attn_output)
            x = self.layer_norms2[i](attn_output + ffn_output)
        
        # Final projection
        output = self.output_proj(x)
        
        return output


class EfficientGraphormerEncoder(nn.Module):
    """
    More efficient Graphormer encoder that processes multiple graphs in parallel.
    This version minimizes the per-graph processing overhead.
    """
    def __init__(self, input_dim, hidden_dim, num_heads=8, num_layers=2, max_dist=20, dropout=0.1):
        super(EfficientGraphormerEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_dist = max_dist
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Spatial encoding (learnable distance embeddings)
        self.spatial_encoding = nn.Embedding(max_dist + 1, hidden_dim)
        
        # Degree encoding
        self.degree_encoding = nn.Embedding(1000, hidden_dim)
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # Feed-forward networks
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms1 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.layer_norms2 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        
        # Final projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def compute_graph_features_batch(self, adj_matrix, num_nodes_list):
        """
        Compute graph structural features for multiple graphs efficiently.
        
        Args:
            adj_matrix: (batch_size, max_nodes, max_nodes) padded adjacency matrix
            num_nodes_list: List of actual node counts for each graph
            
        Returns:
            spatial_pos: (batch_size, max_nodes, max_nodes) spatial positions
            degree_enc: (batch_size, max_nodes, hidden_dim) degree encodings
        """
        batch_size, max_nodes, _ = adj_matrix.shape
        device = adj_matrix.device
        
        # Initialize tensors
        spatial_pos = torch.full((batch_size, max_nodes, max_nodes), self.max_dist, 
                               dtype=torch.long, device=device)
        degree_enc = torch.zeros(batch_size, max_nodes, self.hidden_dim, device=device)
        
        # Process each graph
        for b in range(batch_size):
            num_nodes = num_nodes_list[b]
            if num_nodes == 0:
                continue
                
            # Extract actual graph
            adj = adj_matrix[b, :num_nodes, :num_nodes]
            adj_np = adj.detach().cpu().numpy()
            G = nx.from_numpy_array(adj_np)
            
            # Compute shortest path distances
            try:
                lengths = dict(nx.all_pairs_shortest_path_length(G, cutoff=self.max_dist))
                for i in range(num_nodes):
                    for j in range(num_nodes):
                        if j in lengths.get(i, {}):
                            spatial_pos[b, i, j] = min(lengths[i][j], self.max_dist)
            except:
                pass
            
            # Compute degrees
            degrees = dict(G.degree())
            for i in range(num_nodes):
                if i in degrees:
                    degree_val = min(degrees[i], 999)
                    degree_enc[b, i] = self.degree_encoding(torch.tensor(degree_val, device=device))
        
        return spatial_pos, degree_enc
    
    def forward(self, node_features, adj_matrix, num_nodes_list=None):
        """
        Forward pass with efficient batch processing.
        
        Args:
            node_features: (batch_size, max_nodes, input_dim) padded node features
            adj_matrix: (batch_size, max_nodes, max_nodes) padded adjacency matrix
            num_nodes_list: List of actual node counts for each graph
            
        Returns:
            output: (batch_size, max_nodes, hidden_dim) encoded node features
        """
        batch_size, max_nodes, _ = node_features.shape
        
        # Compute graph structural features
        if num_nodes_list is None:
            # Fallback to individual processing
            spatial_pos, in_degree, out_degree = self.compute_graph_features(adj_matrix)
            degree_enc = self.degree_encoding(in_degree) + self.degree_encoding(out_degree)
        else:
            spatial_pos, degree_enc = self.compute_graph_features_batch(adj_matrix, num_nodes_list)
        
        # Input projection
        x = self.input_proj(node_features)  # (batch_size, max_nodes, hidden_dim)
        
        # Add structural encodings
        spatial_enc = self.spatial_encoding(spatial_pos.view(-1)).view(batch_size, max_nodes, max_nodes, -1)
        
        # Add degree encoding to node features
        x = x + degree_enc
        
        # Apply transformer layers
        for i in range(self.num_layers):
            # Self-attention with spatial bias
            attn_input = x + spatial_enc.mean(dim=2)  # Average spatial encoding across nodes
            
            # Multi-head attention
            attn_output, _ = self.attention_layers[i](attn_input, attn_input, attn_input)
            attn_output = self.layer_norms1[i](x + attn_output)
            
            # Feed-forward network
            ffn_output = self.ffn_layers[i](attn_output)
            x = self.layer_norms2[i](attn_output + ffn_output)
        
        # Final projection
        output = self.output_proj(x)
        
        return output


class ConditionalGraphGenerator(nn.Module):
    def __init__(self, hidden_dim=128, output_dim=2, num_heads=8, num_layers=2):
        super(ConditionalGraphGenerator, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Graphormer encoder to process the input layout and adjacency matrix
        self.graphormer = EfficientGraphormerEncoder(
            input_dim=2,  # x, y coordinates
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )
        
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
        max_num_nodes = input_layout.size(1)
        
        # Convert num_nodes to list if it's a tensor
        if torch.is_tensor(num_nodes):
            num_nodes_list = num_nodes.tolist()
        else:
            num_nodes_list = num_nodes
        
        # Pad all graphs to the same size for batch processing
        padded_layouts = []
        padded_adjs = []
        
        for i in range(batch_size):
            num_nodes_i = num_nodes_list[i] if isinstance(num_nodes_list[i], int) else num_nodes_list[i].item()
            
            # Extract actual data
            layout = input_layout[i][:num_nodes_i, :]
            adj = adj_matrix[i][:num_nodes_i, :num_nodes_i]
            
            # Pad to max size
            layout_padded = F.pad(layout, (0, 0, 0, max_num_nodes - num_nodes_i), value=0)
            adj_padded = F.pad(adj, (0, max_num_nodes - num_nodes_i, 0, max_num_nodes - num_nodes_i), value=0)
            
            padded_layouts.append(layout_padded)
            padded_adjs.append(adj_padded)
        
        # Stack into batch tensors
        batch_layouts = torch.stack(padded_layouts)  # (batch_size, max_num_nodes, 2)
        batch_adjs = torch.stack(padded_adjs)       # (batch_size, max_num_nodes, max_num_nodes)
        
        # Process all graphs through Graphormer encoder
        node_features = self.graphormer(batch_layouts, batch_adjs, num_nodes_list)  # (batch_size, max_num_nodes, hidden_dim)
        
        # Process each graph individually for noise conditioning and output generation
        generated_layouts = []
        for i in range(batch_size):
            num_nodes_i = num_nodes_list[i] if isinstance(num_nodes_list[i], int) else num_nodes_list[i].item()
            
            # Extract features for actual nodes
            features_i = node_features[i][:num_nodes_i, :]  # (num_nodes_i, hidden_dim)
            
            # Encode the noise vector and repeat it to match num_nodes_i
            z_encoding = torch.relu(self.fc_noise(z[i])).unsqueeze(0).repeat(num_nodes_i, 1)

            # Concatenate node features and noise encoding
            conditioned_input = torch.cat([features_i, z_encoding], dim=-1)  # (num_nodes_i, hidden_dim * 2)

            # Generate new node coordinates
            generated_layout = self.fc_out(conditioned_input)  # (num_nodes_i, output_dim)

            # Pad to max size
            padded_layout = F.pad(generated_layout, (0, 0, 0, max_num_nodes - num_nodes_i), value=0)
            generated_layouts.append(padded_layout)

        # Stack all generated layouts
        return torch.stack(generated_layouts)


class GraphDiscriminator(nn.Module):
    def __init__(self, hidden_dim=128, embedding_dim=256, num_heads=8, num_layers=2):
        super(GraphDiscriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        # Graphormer encoder for feature extraction
        self.graphormer = EfficientGraphormerEncoder(
            input_dim=2,  # x, y coordinates
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )
        
        # Projection to embedding dimension
        self.embedding_proj = nn.Linear(hidden_dim, embedding_dim)
        
        # Adversarial head for real/fake layout scoring
        self.adv_head = nn.Linear(embedding_dim, 1)
        
        # Additional layers for better feature extraction
        self.fc1 = nn.Linear(embedding_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def encode_layout(self, input_layout, adj_matrix):
        """
        Shared feature encoder using Graphormer that maps layouts into embedding vectors.
        
        Arguments:
        - input_layout: Node coordinates (batch_size, num_nodes, 2)
        - adj_matrix: Adjacency matrix (batch_size, num_nodes, num_nodes)
        
        Returns:
        - embeddings: Layout embeddings (batch_size, embedding_dim)
        """
        batch_size = input_layout.size(0)
        max_num_nodes = input_layout.size(1)
        
        # Get actual node counts for each graph
        num_nodes_list = []
        for i in range(batch_size):
            # Count non-zero nodes (assuming padded nodes have zero coordinates)
            non_zero_mask = torch.norm(input_layout[i], dim=1) > 0
            num_nodes = non_zero_mask.sum().item()
            num_nodes_list.append(num_nodes)
        
        # Pad all graphs to the same size for batch processing
        padded_layouts = []
        padded_adjs = []
        
        for i in range(batch_size):
            num_nodes_i = num_nodes_list[i]
            
            # Extract actual data
            layout = input_layout[i][:num_nodes_i, :]
            adj = adj_matrix[i][:num_nodes_i, :num_nodes_i]
            
            # Pad to max size
            layout_padded = F.pad(layout, (0, 0, 0, max_num_nodes - num_nodes_i), value=0)
            adj_padded = F.pad(adj, (0, max_num_nodes - num_nodes_i, 0, max_num_nodes - num_nodes_i), value=0)
            
            padded_layouts.append(layout_padded)
            padded_adjs.append(adj_padded)
        
        # Stack into batch tensors
        batch_layouts = torch.stack(padded_layouts)
        batch_adjs = torch.stack(padded_adjs)
        
        # Process all graphs through Graphormer encoder
        node_features = self.graphormer(batch_layouts, batch_adjs, num_nodes_list)  # (batch_size, max_num_nodes, hidden_dim)
        
        # Project to embedding dimension
        node_features = self.embedding_proj(node_features)  # (batch_size, max_num_nodes, embedding_dim)
        
        # Global pooling and MLP processing for each graph
        embeddings = []
        for i in range(batch_size):
            num_nodes_i = num_nodes_list[i]
            
            # Extract features for actual nodes
            features_i = node_features[i][:num_nodes_i, :]  # (num_nodes_i, embedding_dim)
            
            # Global pooling (mean over nodes)
            pooled_features = torch.mean(features_i, dim=0)  # (embedding_dim,)
            
            # Additional MLP layers for better feature representation
            pooled_features = torch.relu(self.fc1(pooled_features))
            pooled_features = self.dropout(pooled_features)
            pooled_features = torch.relu(self.fc2(pooled_features))
            
            embeddings.append(pooled_features)
        
        # Stack all embeddings
        return torch.stack(embeddings)  # (batch_size, embedding_dim)

    def forward(self, input_layout, adj_matrix):
        """
        Forward pass that returns both adversarial scores and embeddings.
        
        Arguments:
        - input_layout: Node coordinates (batch_size, num_nodes, 2)
        - adj_matrix: Adjacency matrix (batch_size, num_nodes, num_nodes)

        Returns:
        - validity_scores: Real/fake scores (batch_size, 1)
        - embeddings: Layout embeddings (batch_size, embedding_dim)
        """
        # Get embeddings using shared encoder
        embeddings = self.encode_layout(input_layout, adj_matrix)
        
        # Adversarial head for real/fake scoring
        validity_scores = self.adv_head(embeddings)
        
        return validity_scores, embeddings


# Utility functions for graph processing
def pad_graph_batch(layouts, adj_matrices, num_nodes_list):
    """
    Pad a batch of graphs to the same size for efficient processing.
    
    Args:
        layouts: List of (num_nodes_i, 2) coordinate tensors
        adj_matrices: List of (num_nodes_i, num_nodes_i) adjacency matrices
        num_nodes_list: List of actual node counts
        
    Returns:
        padded_layouts: (batch_size, max_nodes, 2) padded coordinate tensor
        padded_adjs: (batch_size, max_nodes, max_nodes) padded adjacency tensor
    """
    batch_size = len(layouts)
    max_nodes = max(num_nodes_list)
    
    padded_layouts = []
    padded_adjs = []
    
    for i in range(batch_size):
        num_nodes = num_nodes_list[i]
        
        # Pad layout
        layout_padded = F.pad(layouts[i], (0, 0, 0, max_nodes - num_nodes), value=0)
        padded_layouts.append(layout_padded)
        
        # Pad adjacency matrix
        adj_padded = F.pad(adj_matrices[i], (0, max_nodes - num_nodes, 0, max_nodes - num_nodes), value=0)
        padded_adjs.append(adj_padded)
    
    return torch.stack(padded_layouts), torch.stack(padded_adjs)


def create_attention_mask(num_nodes_list, max_nodes, device):
    """
    Create attention mask for variable-sized graphs.
    
    Args:
        num_nodes_list: List of actual node counts
        max_nodes: Maximum number of nodes in the batch
        device: Device to create tensor on
        
    Returns:
        attention_mask: (batch_size, max_nodes) boolean mask where True = valid node
    """
    batch_size = len(num_nodes_list)
    attention_mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool, device=device)
    
    for i, num_nodes in enumerate(num_nodes_list):
        attention_mask[i, :num_nodes] = True
    
    return attention_mask



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

    # # load B4 380×380
    # model = timm.create_model('efficientnet_b4', pretrained=True)

    # Replace classifier head (nn.Linear) for binary classification
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 2)

    # # #load B6 528×528
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

    # vit = timm.create_model('vit_base_patch16_384', pretrained=True)
    
    # Get the number of input features for the classifier head
    num_ftrs = vit.head.in_features
    
    # Modify the head to classify into 2 classes (Hamiltonian vs Non-Hamiltonian)
    vit.head = nn.Linear(num_ftrs, 2)
    
    return vit
