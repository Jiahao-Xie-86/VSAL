import os
import random
import numpy as np
import networkx as nx

def generate_tree_graph(n, seed=None):
    """
    A uniform random tree on n nodes (exactly n-1 edges).
    """
    return nx.random_tree(n, seed=seed)

def generate_easily_distinguished_nontree_graph(n, seed=None):
    """
    Start with a random tree, then add MANY extra edges:
      • extra = random between 10% and 30% of (n-1)
    This yields a connected graph with (n-1 + extra) edges,
    making it far from a tree and very easy to classify by edge-count.
    """
    if seed is not None:
        random.seed(seed)

    G = nx.random_tree(n, seed=seed)
    non_edges = list(nx.non_edges(G))

    # decide how many extra edges to add: 10%–30% of (n-1)
    base = n - 1
    k_min = max(1, int(0.05 * base))
    k_max = max(1, int(0.25 * base))
    k = random.randint(k_min, k_max)

    # sample and add them
    extras = random.sample(non_edges, k)
    G.add_edges_from(extras)

    return G

def save_graphs(num_graphs=2000,
                n_min=901, n_max=1000,
                tree_dir="tree_graphs",
                nontree_dir="nontree_graphs",
                seed=42):
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(tree_dir, exist_ok=True)
    os.makedirs(nontree_dir, exist_ok=True)

    for i in range(num_graphs):
        n = random.randint(n_min, n_max)

        # --- Tree ---
        T = generate_tree_graph(n, seed=seed + i)
        A_T = nx.to_numpy_array(T, nodelist=range(n), dtype=np.int8)
        np.save(os.path.join(tree_dir, f"tree_{i:04d}.npy"), A_T)

        # --- “Easy” Non-Tree ---
        NT = generate_easily_distinguished_nontree_graph(n, seed=seed + num_graphs + i)
        A_NT = nx.to_numpy_array(NT, nodelist=range(n), dtype=np.int8)
        np.save(os.path.join(nontree_dir, f"nontree_{i:04d}.npy"), A_NT)

        if (i+1) % 100 == 0:
            print(f"Generated {i+1}/{num_graphs} of each")

if __name__ == "__main__":
    save_graphs()
    print("Done generating 2 000 trees and 2 000 non-trees.")
