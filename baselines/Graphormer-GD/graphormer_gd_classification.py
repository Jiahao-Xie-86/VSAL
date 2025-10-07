import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import AdamW
from tqdm import tqdm
import networkx as nx
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import Counter
import random
import argparse
from typing import Dict, List, Tuple, Optional
import warnings

# Add Graphormer-GD to path
sys.path.append('./Graphormer-GD')

from graphormer.data.collator import collator_with_resistance_distance
# Remove OGB and wrapper imports
# from graphormer.data.wrapper import preprocess_item_with_resistance_distance

def safe_nan_to_num(tensor, nan=512.0, posinf=512.0, neginf=512.0):
    """Safely convert NaN/inf values in tensor to finite numbers."""
    if torch.is_tensor(tensor):
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            warnings.warn(f"NaN or inf values detected in tensor, replacing with {nan}")
            return torch.nan_to_num(tensor, nan=nan, posinf=posinf, neginf=neginf)
        return tensor
    elif isinstance(tensor, np.ndarray):
        if np.isnan(tensor).any() or np.isinf(tensor).any():
            warnings.warn(f"NaN or inf values detected in numpy array, replacing with {nan}")
            return np.nan_to_num(tensor, nan=nan, posinf=posinf, neginf=neginf)
        return tensor
    else:
        return tensor

# Add local versions of required functions
import numpy as np
import torch
from torch_geometric.data import Data

def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x

def floyd_warshall(adjacency_matrix):
    n = adjacency_matrix.shape[0]
    M = adjacency_matrix.astype(np.int64)
    path = -1 * np.ones([n, n], dtype=np.int64)
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i][j] = 0
            elif M[i][j] == 0:
                M[i][j] = 510
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if M[i][j] > M[i][k] + M[k][j]:
                    M[i][j] = M[i][k] + M[k][j]
                    path[i][j] = k
    for i in range(n):
        for j in range(n):
            if M[i][j] >= 510:
                path[i][j] = 510
                M[i][j] = 510
    return M, path

def get_all_edges(path, i, j):
    k = path[i][j]
    if k == -1:
        return []
    else:
        return get_all_edges(path, i, k) + [k] + get_all_edges(path, k, j)

def gen_edge_input(max_dist, path, edge_feat):
    n = path.shape[0]
    edge_fea_all = -1 * np.ones([n, n, max_dist, edge_feat.shape[-1]], dtype=np.int64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if path[i][j] == 510:
                continue
            p = [i] + get_all_edges(path, i, j) + [j]
            num_path = len(p) - 1
            for k in range(num_path):
                edge_fea_all[i, j, k, :] = edge_feat[p[k], p[k+1], :]
    return edge_fea_all

def preprocess_item_with_resistance_distance(item):
    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x
    resistance_distance = item.res_pos
    N = x.size(0)
    x = convert_to_single_emb(x)
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
        convert_to_single_emb(edge_attr) + 1
    )
    shortest_path_result, path = floyd_warshall(adj.numpy())
    max_dist = np.amax(shortest_path_result)
    edge_input = gen_edge_input(max_dist, path, attn_edge_type.numpy())
    spatial_pos = torch.from_numpy((shortest_path_result)).long()
    res_pos = resistance_distance[:, :N]
    attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token
    item.x = x
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.spatial_pos = spatial_pos
    item.res_pos = res_pos
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = item.in_degree  # for undirected graph
    item.edge_input = torch.from_numpy(edge_input).long()
    return item

from graphormer.models.graphormer_gd import GraphormerGDModel
from graphormer.models.graphormer_gd import GraphormerGDEncoder
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.configs import FairseqConfig
from dataclasses import dataclass, field
from omegaconf import DictConfig, OmegaConf

# Add the train function here

def train(model, train_loader, val_loader, test_loader, device, optimizer, criterion, scheduler, num_epochs, patience):
    """Train the model with early stopping and log results to CSV."""
    best_val_loss = float('inf')
    best_accuracy = 0
    early_stop_counter = 0
    best_model_path = './best_graphormer_gd_model.pth'
    columns = ['epoch', 'train_loss', 'train_acc', 'train_f1', 'val_loss', 'val_acc', 'val_f1', 'test_loss', 'test_acc', 'test_f1']
    results = []

    # Initial evaluation before training
    initial_train_loss, initial_train_acc, initial_train_f1 = evaluate(model, train_loader, device, criterion)
    initial_val_loss, initial_val_acc, initial_val_f1 = evaluate(model, val_loader, device, criterion)
    initial_test_loss, initial_test_acc, initial_test_f1 = evaluate(model, test_loader, device, criterion)

    # Log initial metrics before the first epoch
    results.append([0, initial_train_loss, initial_train_acc, initial_train_f1, initial_val_loss, initial_val_acc, initial_val_f1, initial_test_loss, initial_test_acc, initial_test_f1])
    df = pd.DataFrame(results, columns=columns)
    df.to_csv('graphormer_gd_training_results.csv', index=False)

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss, correct, total = 0, 0, 0
        all_preds, all_labels = [], []
        valid_batches = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}')):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            labels = batch['y']  # shape [batch_size]
            optimizer.zero_grad()
            try:
                # Fix NaN values in input tensors
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor) and torch.isnan(v).any():
                        print(f"NaN detected in batch tensor {k}, replacing with 512.0")
                        batch[k] = torch.nan_to_num(v, nan=512.0, posinf=512.0, neginf=512.0)

                outputs = model(batch)
                logits = outputs[:, 0, :]
                
                # Skip batch if logits contain NaN
                if torch.isnan(logits).any():
                    print(f"NaN detected in logits for batch {batch_idx}, skipping")
                    continue

                loss = criterion(logits, labels)
                
                # Skip batch if loss is NaN
                if torch.isnan(loss):
                    print(f"NaN detected in loss for batch {batch_idx}, skipping")
                    continue

                loss.backward()

                # Check for NaN gradients and skip if found
                has_nan_grad = False
                for param in model.parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        has_nan_grad = True
                        break
                
                if has_nan_grad:
                    print(f"NaN gradients detected in batch {batch_idx}, skipping")
                    optimizer.zero_grad()
                    continue

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                _, preds = torch.max(logits, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(preds.cpu())
                all_labels.extend(labels.cpu())
                valid_batches += 1
            except Exception as e:
                print(f"Error in training batch {batch_idx}: {e}")
                continue

        if valid_batches == 0:
            print(f"Warning: No valid batches in epoch {epoch}")
            avg_train_loss = float('nan')
            train_acc = 0.0
            train_f1 = 0.0
        else:
            train_acc, train_f1 = compute_metrics(torch.tensor(all_preds), torch.tensor(all_labels))
            avg_train_loss = train_loss / valid_batches

        # Evaluate on validation set
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device, criterion)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)

        # Early stopping
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            early_stop_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with validation accuracy: {val_acc:.2f}%")
        else:
            early_stop_counter += 1
            print(f"Epochs without improvement: {early_stop_counter}")
        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            return

        # Test evaluation
        test_loss, test_acc, test_f1 = evaluate(model, test_loader, device, criterion)

        # Log results
        results.append([epoch, avg_train_loss, train_acc, train_f1, val_loss, val_acc, val_f1, test_loss, test_acc, test_f1])
        df = pd.DataFrame(results, columns=columns)
        df.to_csv('graphormer_gd_training_results.csv', index=False)
        print(f"Epoch {epoch}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}")

    print("Training complete. Best model saved as best_graphormer_gd_model.pth")

# Function to set random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def check_tensor_validity(tensor, name):
    return True  # No-op for production

@dataclass
class GraphormerGDConfig(FairseqDataclass):
    # Model architecture
    encoder_embed_dim: int = field(default=768, metadata={"help": "encoder embedding dimension"})
    encoder_ffn_embed_dim: int = field(default=768, metadata={"help": "encoder embedding dimension for FFN"})
    encoder_layers: int = field(default=12, metadata={"help": "num encoder layers"})
    encoder_attention_heads: int = field(default=32, metadata={"help": "num encoder attention heads"})
    
    # Graph-specific parameters
    num_atoms: int = field(default=512, metadata={"help": "number of atom types"})
    num_in_degree: int = field(default=512, metadata={"help": "number of in degree types"})
    num_out_degree: int = field(default=512, metadata={"help": "number of out degree types"})
    num_edges: int = field(default=1536, metadata={"help": "number of edge types"})
    num_spatial: int = field(default=512, metadata={"help": "number of spatial types"})
    num_edge_dis: int = field(default=128, metadata={"help": "number of edge distance types"})
    edge_type: str = field(default="multi_hop", metadata={"help": "edge type"})
    multi_hop_max_dist: int = field(default=20, metadata={"help": "max distance for multi-hop edges"})
    max_nodes: int = field(default=512, metadata={"help": "max number of nodes"})
    
    # Graphormer-GD specific parameters
    no_node_feature: bool = field(default=False, metadata={"help": "do not use node feature"})
    no_edge_feature: bool = field(default=False, metadata={"help": "do not use edge feature"})
    num_rd_bias_kernel: int = field(default=128, metadata={"help": "number of kernel function for resistance distance"})
    no_share_bias: bool = field(default=False, metadata={"help": "do not share attention bias"})
    relu_mul_bias: bool = field(default=False, metadata={"help": "use relu function for attn_bias_mul"})
    one_init_mul_bias: bool = field(default=False, metadata={"help": "initiate attn_bias_mul by constant 1.0"})
    mul_bias_with_edge_feature: bool = field(default=False, metadata={"help": "use edge feature in mul bias"})
    droppath_prob: float = field(default=0.0, metadata={"help": "droppath probability"})
    
    # Dropout parameters
    dropout: float = field(default=0.0, metadata={"help": "dropout probability"})
    attention_dropout: float = field(default=0.1, metadata={"help": "dropout probability for attention weights"})
    act_dropout: float = field(default=0.1, metadata={"help": "dropout probability after activation in FFN"})
    
    # Other parameters
    encoder_normalize_before: bool = field(default=False, metadata={"help": "apply layernorm before each encoder block"})
    pre_layernorm: bool = field(default=False, metadata={"help": "apply layernorm before self-attention and ffn"})
    activation_fn: str = field(default="gelu", metadata={"help": "activation function to use"})
    apply_graphormer_init: bool = field(default=False, metadata={"help": "use custom param initialization for Graphormer"})
    share_encoder_input_output_embed: bool = field(default=False, metadata={"help": "share input and output embeddings in the encoder"})
    
    # Classification parameters
    num_classes: int = field(default=2, metadata={"help": "number of classes for classification"})
    
    # Training parameters
    batch_size: int = field(default=16, metadata={"help": "batch size"})
    learning_rate: float = field(default=1e-5, metadata={"help": "learning rate"})
    num_epochs: int = field(default=100, metadata={"help": "number of epochs"})
    patience: int = field(default=10, metadata={"help": "early stopping patience"})

class GraphDataset(Dataset):
    def __init__(self, hamiltonian_dir, non_hamiltonian_dir):
        self.graphs = []
        self.labels = []
        self.datas = []

        idx = 0
        # Load Hamiltonian graphs (label = 1)
        for filename in os.listdir(hamiltonian_dir):
            if filename.endswith('.npy'):
                adj_matrix = np.load(os.path.join(hamiltonian_dir, filename))
                label = 1
                data = self.adj_to_data(adj_matrix, label, idx)
                self.datas.append(data)
                idx += 1

        # Load Non-Hamiltonian graphs (label = 0)
        for filename in os.listdir(non_hamiltonian_dir):
            if filename.endswith('.npy'):
                adj_matrix = np.load(os.path.join(non_hamiltonian_dir, filename))
                label = 0
                data = self.adj_to_data(adj_matrix, label, idx)
                self.datas.append(data)
                idx += 1

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        return self.datas[idx]

    @staticmethod
    def adj_to_data(adj_matrix, label, idx):
        """Convert adjacency matrix and label to a PyG Data object with resistance distance and idx."""
        # Convert adjacency matrix to edge_index
        edge_index = torch.tensor(np.array(np.nonzero(adj_matrix)), dtype=torch.long)
        N = adj_matrix.shape[0]
        # Node features: degree as feature
        degrees = np.sum(adj_matrix, axis=1, keepdims=True)
        x = torch.tensor(degrees, dtype=torch.long)
        # Edge attributes: all ones (or zeros if you want no edge features)
        edge_attr = torch.ones(edge_index.shape[1], 1, dtype=torch.long)
        # Resistance distance
        g = nx.Graph(adj_matrix)
        g_components_list = [g.subgraph(c).copy() for c in nx.connected_components(g)]
        g_resistance_matrix = np.zeros((N, N)) - 1.0
        g_index = 0
        for item in g_components_list:
            cur_adj = nx.to_numpy_array(item)
            cur_num_nodes = cur_adj.shape[0]
            cur_res_dis = np.linalg.pinv(
                np.diag(cur_adj.sum(axis=-1)) - cur_adj + np.ones((cur_num_nodes, cur_num_nodes),
                                                                  dtype=np.float32) / cur_num_nodes
            ).astype(np.float32)
            # Fix NaNs and infs in cur_res_dis
            cur_res_dis = np.nan_to_num(cur_res_dis, nan=512.0, posinf=512.0, neginf=512.0)
            A = np.diag(cur_res_dis)[:, None]
            B = np.diag(cur_res_dis)[None, :]
            cur_res_dis = A + B - 2 * cur_res_dis
            g_resistance_matrix[g_index:g_index + cur_num_nodes, g_index:g_index + cur_num_nodes] = cur_res_dis
            g_index += cur_num_nodes
        g_cur_index = []
        for item in g_components_list:
            g_cur_index.extend(list(item.nodes))
        ori_idx = np.arange(N)
        g_resistance_matrix[g_cur_index, :] = g_resistance_matrix[ori_idx, :]
        g_resistance_matrix[:, g_cur_index] = g_resistance_matrix[:, ori_idx]
        if g_resistance_matrix.max() > N - 1:
            print(f'error: {g_resistance_matrix}')
        g_resistance_matrix[g_resistance_matrix == -1.0] = 512.0
        # Fix NaNs and infs in g_resistance_matrix
        g_resistance_matrix = np.nan_to_num(g_resistance_matrix, nan=512.0, posinf=512.0, neginf=512.0)
        res_matrix = np.zeros((N, N), dtype=np.float32)
        res_matrix[:, :N] = g_resistance_matrix
        res_pos = torch.from_numpy(res_matrix).float()
        # Create PyG Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor(label, dtype=torch.long), res_pos=res_pos)
        data.idx = idx
        # Use official preprocessing
        data = preprocess_item_with_resistance_distance(data)
        return data

def collate_fn(batch):
    """Custom collate function for Graphormer-GD"""
    return collator_with_resistance_distance(batch, max_node=512, multi_hop_max_dist=20, spatial_pos_max=100)

def compute_metrics(preds, labels):
    """Compute accuracy and F1-score"""
    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return accuracy, f1

def validate_batch(batch, device):
    """Validate batch tensors for NaN/inf values"""
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if not check_tensor_validity(value, key):
                print(f"Invalid tensor found in batch key: {key}")
                print(f"Tensor shape: {value.shape}")
                print(f"Tensor dtype: {value.dtype}")
                print(f"Tensor min/max: {value.min().item()}/{value.max().item()}")
                return False
    return True

def evaluate(model, dataloader, device, criterion):
    """Evaluate the model and return average loss, accuracy, and F1-score."""
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    valid_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            labels = batch['y']  # shape [batch_size]

            try:
                # Final NaN check and fix before model forward
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor) and torch.isnan(v).any():
                        print(f"NaN detected in batch tensor {k}, replacing with 512.0")
                        batch[k] = torch.nan_to_num(v, nan=512.0, posinf=512.0, neginf=512.0)

                outputs = model(batch)
                logits = outputs[:, 0, :]
                loss = criterion(logits, labels)
                
                total_loss += loss.item()
                _, preds = torch.max(logits, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(preds.cpu())
                all_labels.extend(labels.cpu())
                valid_batches += 1
            except Exception as e:
                print(f"Error in evaluation batch {batch_idx}: {e}")
                continue

    if valid_batches == 0:
        print("Warning: No valid batches in evaluation")
        return float('nan'), 0.0, 0.0
    
    avg_loss = total_loss / valid_batches
    accuracy, f1 = compute_metrics(torch.tensor(all_preds), torch.tensor(all_labels))
    
    return avg_loss, accuracy, f1

def initialize_model_parameters(model):
    """Initialize model parameters to prevent NaN issues."""
    for name, param in model.named_parameters():
        if param.requires_grad:
            if len(param.shape) > 1:
                # Use Xavier/Glorot initialization for weight matrices
                torch.nn.init.xavier_uniform_(param, gain=0.1)
            else:
                # Initialize biases to zero
                torch.nn.init.zeros_(param)
    print("Model parameters initialized")

def create_model(config):
    """Create Graphormer-GD model"""
    # Create a simple args-like object
    class Args:
        def __init__(self, config):
            for key, value in config.items():
                setattr(self, key, value)

    args = Args(config)

    # Create the encoder (must be a FairseqEncoder)
    encoder = GraphormerGDEncoder(args)

    # Create the model with the encoder
    model = GraphormerGDModel(args, encoder)

    return model

def test_single_batch(model, train_loader, device):
    """Test a single batch to isolate NaN issues."""
    model.eval()
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx > 0:  # Only test first batch
            break
            
        print(f"Testing batch {batch_idx}")
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Check all input tensors
        print("Input tensor stats:")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={v.shape}, min={v.min().item()}, max={v.max().item()}, any_nan={torch.isnan(v).any().item()}")
        
        # Fix any NaN values
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and torch.isnan(v).any():
                print(f"Fixing NaN in {k}")
                batch[k] = torch.nan_to_num(v, nan=512.0, posinf=512.0, neginf=512.0)
        
        # Test model forward pass
        try:
            with torch.no_grad():
                outputs = model(batch)
                logits = outputs[:, 0, :]
                print(f"Model output stats: min={logits.min().item()}, max={logits.max().item()}, any_nan={torch.isnan(logits).any().item()}")
                
                if torch.isnan(logits).any():
                    print("ERROR: Model is producing NaN outputs!")
                    return False
                else:
                    print("SUCCESS: Model produces valid outputs")
                    return True
        except Exception as e:
            print(f"ERROR in model forward pass: {e}")
            return False

def main():
    # Set the random seed
    set_seed(42)
    
    # Configuration
    config = {
        'encoder_embed_dim': 256,  # Reduced from 768
        'encoder_ffn_embed_dim': 256,  # Reduced from 768
        'encoder_layers': 1,  # Reduced to minimal for stability
        'encoder_attention_heads': 4,  # Reduced from 12
        'num_atoms': 512,
        'num_in_degree': 512,
        'num_out_degree': 512,
        'num_edges': 1536,
        'num_spatial': 512,
        'num_edge_dis': 128,
        'edge_type': 'multi_hop',
        'multi_hop_max_dist': 20,
        'max_nodes': 512,
        'no_node_feature': False,
        'no_edge_feature': False,
        'num_rd_bias_kernel': 128,
        'no_share_bias': False,
        'relu_mul_bias': False,
        'one_init_mul_bias': False,
        'mul_bias_with_edge_feature': False,
        'droppath_prob': 0.0,
        'dropout': 0.0,
        'attention_dropout': 0.0,  # Reduced from 0.1
        'act_dropout': 0.0,  # Reduced from 0.1
        'encoder_normalize_before': False,
        'pre_layernorm': False,
        'activation_fn': 'gelu',
        'apply_graphormer_init': False,
        'share_encoder_input_output_embed': False,
        'pretrained_model_name': 'none',
        'num_classes': 2,
        'batch_size': 16,  # Reduced from 16
        'learning_rate': 1e-5,  
        'num_epochs': 100,
        'patience': 10
    }
    
    # Directories containing the .npy files
    hamiltonian_dir = './new_401-500_planar_graphs_dataset/planar_graphs'
    non_hamiltonian_dir = './new_401-500_planar_graphs_dataset/nonplanar_graphs'
    
    # Initialize dataset
    dataset = GraphDataset(hamiltonian_dir, non_hamiltonian_dir)
    indices = list(range(len(dataset)))
    
    # Specify the train_val size and test size
    train_val_size = 1000
    test_size = 500
    
    # Split dataset into train_val and test sets
    train_val_indices, test_indices = train_test_split(
        indices, test_size=test_size, 
        stratify=[dataset[i].y.item() for i in indices], random_state=41
    )
    
    # Split train_val_indices into train and validation sets (80% train, 20% validation)
    train_indices, val_indices = train_test_split(
        train_val_indices[:train_val_size], test_size=0.2, 
        stratify=[dataset[i].y.item() for i in train_val_indices[:train_val_size]], random_state=41
    )
    
    # Create subsets for train, validation, and test datasets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    # Calculate the distribution of training, validation, and test sets
    train_counter = Counter([dataset[i].y.item() for i in train_indices])
    val_counter = Counter([dataset[i].y.item() for i in val_indices])
    test_counter = Counter([dataset[i].y.item() for i in test_indices])
    
    # Print the class distribution
    print(f"Training Dataset: {train_counter[1]} True, {train_counter[0]} False")
    print(f"Validation Dataset: {val_counter[1]} True, {val_counter[0]} False")
    print(f"Test Dataset: {test_counter[1]} True, {test_counter[0]} False")
    
    num_workers = os.cpu_count()
    # Define data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
    
    # Create model
    model = create_model(config)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Print model info
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss()
    
    # Test a single batch to isolate NaN issues
    test_single_batch(model, train_loader, device)

    # Train the model
    train(model, train_loader, val_loader, test_loader, device, optimizer, criterion, scheduler,
          config['num_epochs'], config['patience'])

if __name__ == "__main__":
    main() 