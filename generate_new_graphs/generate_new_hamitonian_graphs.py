import os
import numpy as np
import networkx as nx
from concurrent.futures import ProcessPoolExecutor, as_completed
import random

def make_hamiltonian_graph(n_nodes, p_extra):
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    cycle = [(i, (i+1) % n_nodes) for i in range(n_nodes)]
    G.add_edges_from(cycle)
    for i in range(n_nodes):
        for j in range(i+2, n_nodes):
            if i == 0 and j == n_nodes-1:
                continue
            if np.random.rand() < p_extra:
                G.add_edge(i, j)
    return nx.to_numpy_array(G, dtype=np.int8)

def make_nonhamiltonian_graph(n_nodes, p_extra, min_breaks=200, max_breaks=300):
    """
    Start from the n‑cycle, then remove between min_breaks and max_breaks edges,
    then sprinkle on extra chords with probability p_extra.
    This almost certainly breaks *any* Hamiltonian cycle, making the class much easier.
    """
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))

    # 1) create the base cycle
    cycle = [(i, (i+1) % n_nodes) for i in range(n_nodes)]
    G.add_edges_from(cycle)

    # 2) remove multiple cycle edges
    k = random.randint(min_breaks, max_breaks)
    to_remove = random.sample(cycle, k)
    G.remove_edges_from(to_remove)

    # 3) add random extra edges, but *not* the ones we removed
    for i in range(n_nodes):
        for j in range(i+2, n_nodes):
            # skip the wrap‑around back‑edge already in the cycle
            if i == 0 and j == n_nodes-1:
                continue
            # and skip if it's one of the removed cycle edges
            if (i, j) in to_remove or (j, i) in to_remove:
                continue
            if np.random.rand() < p_extra:
                G.add_edge(i, j)

    # Return as int8 adjacency matrix
    return nx.to_numpy_array(G, dtype=np.int8)

def _worker(args):
    idx, n_nodes, p_extra, out_dir, is_ham = args
    A = (make_hamiltonian_graph if is_ham else make_nonhamiltonian_graph)(n_nodes, p_extra)
    sub = "ham" if is_ham else "nonham"
    fname = f"{sub}_{idx:05d}_n{n_nodes}.npy"
    path = os.path.join(out_dir, sub, fname)
    np.save(path, A)
    return path

def generate_dataset(n_graphs_per_class,
                     node_range,
                     p_extra,
                     out_dir="graphs",
                     n_workers=None):
    os.makedirs(os.path.join(out_dir, "ham"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "nonham"), exist_ok=True)

    tasks = []
    for cls in (True, False):
        for i in range(n_graphs_per_class):
            n = np.random.randint(node_range[0], node_range[1] + 1)
            tasks.append((i, n, p_extra, out_dir, cls))

    n_workers = n_workers or os.cpu_count()
    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        for fut in as_completed(exe.submit(_worker, t) for t in tasks):
            _ = fut.result()  # you can print this if you like

if __name__ == "__main__":
    generate_dataset(
        n_graphs_per_class=2000,
        node_range=(901, 1000),
        p_extra=0.005,
        out_dir="./dataset",
    )
