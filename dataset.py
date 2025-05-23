import os
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
import networkx as nx
import time
import concurrent.futures


def generate_good_layouts(adj_matrix, num_variations=10, layout_type='circular'):
    """
    Generates graph layouts focusing on various layout types for ablation experiments.
    Supports circular, shell, random, and spiral layouts, with perturbations for variety.
    
    Args:
    - adj_matrix (np.array): The adjacency matrix of the input graph.
    - num_variations (int): Number of layout variations to generate.
    - layout_type (str): The type of layout to generate ('circular', 'shell', 'random', 'spiral').
    
    Returns:
    - layouts (list of np.array): List of generated graph layouts.
    """


    G = nx.from_numpy_array(adj_matrix)

    layouts = []

    # Define primary layout based on the layout_type
    if layout_type == 'circular':
        primary_layout = nx.circular_layout(G)
    elif layout_type == 'shell':
        primary_layout = nx.shell_layout(G)
    elif layout_type == 'random':
        primary_layout = nx.random_layout(G)
    elif layout_type == 'spiral':
        # Create a custom spiral layout function
        def spiral_layout(G):
            n = len(G.nodes())
            spiral_layout_positions = {}
            for i, node in enumerate(G.nodes()):
                angle = i * 0.2
                radius = 0.2 * i
                spiral_layout_positions[node] = np.array([radius * np.cos(angle), radius * np.sin(angle)])
            return spiral_layout_positions
        
        primary_layout = spiral_layout(G)
    else:
        raise ValueError("Invalid layout_type. Choose from 'circular', 'shell', 'random', or 'spiral'.")
    

    # Convert layout to numpy array format
    layouts.append(np.array([primary_layout[node] for node in G.nodes()]))


    # Repulsion mechanism to adjust nodes that are too close
    def apply_repulsion(layout, repulsion_distance=0.1):
        for i in range(len(layout)):
            for j in range(i + 1, len(layout)):
                dist = np.linalg.norm(layout[i] - layout[j])
                if dist < repulsion_distance:
                    direction = (layout[i] - layout[j]) / (dist + 1e-6)
                    force = 0.05 / (dist + 1e-6)
                    layout[i] += direction * force
                    layout[j] -= direction * force
        return layout

    # Edge crossing minimization using spring layout post-optimization
    def minimize_edge_crossings(layout, G):
        pos = {i: layout[i] for i in range(len(layout))}
        spring_layout = nx.spring_layout(G, pos=pos, iterations=100)
        kamada_kawai_layout = nx.kamada_kawai_layout(G, pos=spring_layout)
        return np.array([kamada_kawai_layout[node] for node in G.nodes()])
        # return np.array([spring_layout[node] for node in G.nodes()])

    # Generate additional variations by adding small perturbations to the primary layout
    for _ in range(num_variations):
        perturbed_layout = np.array([primary_layout[node] + np.random.normal(scale=0.005, size=2) for node in G.nodes()])
        perturbed_layout = apply_repulsion(perturbed_layout)
        optimized_layout = minimize_edge_crossings(perturbed_layout, G)
        # optimized_layout = perturbed_layout
        layouts.append(optimized_layout)

    return layouts


# # ---- Custom Dataset for Hamiltonian and Non-Hamiltonian Graph Layouts ----
# class HamiltonianGraphDataset(Dataset):
#     def __init__(self, hamiltonian_dir, non_hamiltonian_dir, pretrain=False):
#         # Load all .npy files from both directories
#         self.hamiltonian_files = [os.path.join(hamiltonian_dir, f) for f in os.listdir(hamiltonian_dir) if f.endswith('.npy')]
#         self.non_hamiltonian_files = [os.path.join(non_hamiltonian_dir, f) for f in os.listdir(non_hamiltonian_dir) if f.endswith('.npy')]
        
#         # Assign labels: 1 for Hamiltonian, 0 for non-Hamiltonian
#         self.labels = [1] * len(self.hamiltonian_files) + [0] * len(self.non_hamiltonian_files)
        
#         # Combine the file lists
#         self.all_files = self.hamiltonian_files + self.non_hamiltonian_files

#         self.pretrain = pretrain

#     def __len__(self):
#         return len(self.all_files)


#     def __getitem__(self, idx):
#         # Load the adjacency matrix from the .npy file
#         adj_matrix = np.load(self.all_files[idx])  # Now, it's just an adjacency matrix

#         # Optionally, you can generate random node coordinates for each graph
#         num_nodes = adj_matrix.shape[0]

#         #node_coords = np.random.rand(num_nodes, 2) * 2 - 1  # Random node coordinates in [-1, 1] range

#         if self.pretrain:
#             # Generate various good layouts for pretraining
#             layouts = generate_good_layouts(adj_matrix)
#             node_coords = layouts[np.random.randint(len(layouts))]  # Choose one randomly
#         else:
#             # Random node coordinates for each graph (for normal training)
#             #node_coords = np.random.rand(num_nodes, 2) * 2 - 1  # Random node coordinates in [-1, 1] range
#             layouts = generate_good_layouts(adj_matrix) # Also use good layouts for training
#             node_coords = layouts[np.random.randint(len(layouts))]  # Choose one randomly
        
#         # Get the corresponding label
#         label = self.labels[idx]
#         return node_coords, adj_matrix, label


# Define a helper function for parallel precomputation.
def compute_layout(file):
    # Load the adjacency matrix from the given file.
    adj_matrix = np.load(file)
    # Compute one layout (you can adjust num_variations if needed)
    layout = generate_good_layouts(adj_matrix, num_variations=10)[0]
    return file, layout

class HamiltonianGraphDataset(Dataset):
    def __init__(self, hamiltonian_dir, non_hamiltonian_dir, pretrain=False, cache_layouts=False):
        self.hamiltonian_files = [os.path.join(hamiltonian_dir, f) for f in os.listdir(hamiltonian_dir) if f.endswith('.npy')]
        self.non_hamiltonian_files = [os.path.join(non_hamiltonian_dir, f) for f in os.listdir(non_hamiltonian_dir) if f.endswith('.npy')]
        self.labels = [1] * len(self.hamiltonian_files) + [0] * len(self.non_hamiltonian_files)
        self.all_files = self.hamiltonian_files + self.non_hamiltonian_files
        self.pretrain = pretrain
        self.cache_layouts = cache_layouts

        # Precompute layouts if caching is enabled
        self.layout_cache = {}
        if self.cache_layouts:
            print("Precomputing layouts in parallel using multiple CPU cores...")
            # for file in self.all_files:
            #     adj_matrix = np.load(file)
            #     # Compute a single layout or a few variations
            #     layout = generate_good_layouts(adj_matrix, num_variations=1)[0] 
            #     self.layout_cache[file] = layout
            with concurrent.futures.ProcessPoolExecutor() as executor:
                # Use map to process all files concurrently.
                results = executor.map(compute_layout, self.all_files)
                for file, layout in results:
                    self.layout_cache[file] = layout
    
    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        file = self.all_files[idx]
        adj_matrix = np.load(file)
        if self.cache_layouts:
            node_coords = self.layout_cache[file]
        else:
            layouts = generate_good_layouts(adj_matrix)
            node_coords = layouts[np.random.randint(len(layouts))]
        label = self.labels[idx]
        return node_coords, adj_matrix, label



# def custom_collate(batch, device):
#     # Extract node coordinates and adjacency matrices
#     node_coords = [torch.tensor(each[0], dtype=torch.float32).to(device) for each in batch]
#     adj_matrices = [torch.tensor(each[1], dtype=torch.float32).to(device) for each in batch]

#     # Determine the maximum size among the adjacency matrices
#     max_size = max(matrix.size(0) for matrix in adj_matrices)

#     # Pad each adjacency matrix and node coordinate matrix to the maximum size
#     padded_coords = [F.pad(coord, (0, 0, 0, max_size - coord.size(0))) for coord in node_coords]
#     padded_adj_matrices = [F.pad(matrix, (0, max_size - matrix.size(0), 0, max_size - matrix.size(1))) for matrix in adj_matrices]

#     # Create batched tensors for both padded coordinates and adjacency matrices
#     coords_batch = torch.stack(padded_coords)
#     adj_matrices_batch = torch.stack(padded_adj_matrices)

#     labels = torch.tensor([each[2] for each in batch], dtype=torch.long).to(device)  # Labels

#     return coords_batch, adj_matrices_batch, labels


def custom_collate(batch):
    # Extract node coordinates and adjacency matrices
    node_coords = [torch.tensor(each[0], dtype=torch.float32) for each in batch]
    adj_matrices = [torch.tensor(each[1], dtype=torch.float32) for each in batch]

    # Determine the maximum size among the adjacency matrices
    max_size = max(matrix.size(0) for matrix in adj_matrices)

    # Pad each adjacency matrix and node coordinate matrix to the maximum size
    padded_coords = [F.pad(coord, (0, 0, 0, max_size - coord.size(0))) for coord in node_coords]
    padded_adj_matrices = [F.pad(matrix, (0, max_size - matrix.size(0), 0, max_size - matrix.size(1))) for matrix in adj_matrices]

    # Create batched tensors for both padded coordinates and adjacency matrices
    coords_batch = torch.stack(padded_coords)
    adj_matrices_batch = torch.stack(padded_adj_matrices)

    labels = torch.tensor([each[2] for each in batch], dtype=torch.long)  # Labels

    return coords_batch, adj_matrices_batch, labels
