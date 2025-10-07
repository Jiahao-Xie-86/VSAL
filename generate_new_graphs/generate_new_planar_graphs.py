#!/usr/bin/env python3
import os
import numpy as np
import networkx as nx
import random, itertools
from multiprocessing import Pool, cpu_count

# ── Configuration ─────────────────────────────────────────────────────────────
NUM_GRAPHS     = 2000
NODE_RANGE     = (401, 500)
DENSITY_FACTOR = 1.6   # planar graphs will have ≈ n-1 + extra edges, extra ≈ 0.6·n
PLANAR_DIR     = "planar_graphs"
NONPLANAR_DIR  = "nonplanar_graphs"

os.makedirs(PLANAR_DIR,   exist_ok=True)
os.makedirs(NONPLANAR_DIR, exist_ok=True)

# ── Helpers ────────────────────────────────────────────────────────────────────
def random_sparse_planar(n, density=DENSITY_FACTOR):
    """
    Start with a random spanning tree (n-1 edges), then add up to
    `extra = int((density - 1.0) * n)` edges at random (only if they preserve planarity).
    """
    # 1) random tree
    G = nx.random_tree(n)
    target_edges = int(density * n)
    extra_needed = target_edges - (n - 1)

    # 2) try adding random edges
    nodes = list(G.nodes())
    attempts = 0
    while extra_needed > 0 and attempts < extra_needed * 10:
        u, v = random.sample(nodes, 2)
        if G.has_edge(u, v):
            attempts += 1
            continue
        G.add_edge(u, v)
        planar, _ = nx.check_planarity(G, False)
        if planar:
            extra_needed -= 1
        else:
            G.remove_edge(u, v)
            attempts += 1

    return G

# def sparse_nonplanar_from_planar(G):
#     """
#     Given a sparse planar G, remove one edge and then add one
#     non-edge that makes the graph non-planar (but keeps |E| constant).
#     """
#     H = G.copy()
#     n = H.number_of_nodes()

#     # remove random edge to make room
#     u_rem, v_rem = random.choice(list(H.edges()))
#     H.remove_edge(u_rem, v_rem)

#     # add forbidden edge
#     nodes = list(H.nodes())
#     while True:
#         u, v = random.sample(nodes, 2)
#         if H.has_edge(u, v):
#             continue
#         H.add_edge(u, v)
#         is_planar, _ = nx.check_planarity(H, False)
#         if not is_planar:
#             break
#         H.remove_edge(u, v)

#     return H

def plant_forbidden_minor(G):
    """
    Take a sparse planar G, choose K5 or K3,3 at random,
    embed it (adding 10 edges for K5 or 9 edges for K3,3),
    then delete the same # of other edges so |E| is constant.
    """
    H = G.copy()
    n = H.number_of_nodes()
    E = list(H.edges())

    # choose which minor to plant
    if random.random() < 0.5 and n >= 5:
        # plant K5
        nodes = random.sample(range(n), 5)
        edges_to_plant = list(itertools.combinations(nodes, 2))  # 10 edges
    elif n >= 6:
        # plant K3,3
        part = random.sample(range(n), 6)
        A, B = part[:3], part[3:]
        edges_to_plant = [(u, v) for u in A for v in B]        # 9 edges
    else:
        # fallback to the “single‐edge trick” if too small
        # (this almost never happens when n >= 901)
        u, v = random.sample(range(n), 2)
        edges_to_plant = [(u, v)]

    k = len(edges_to_plant)

    # remove k random existing edges (but don't remove edges we plan to add)
    removable = [e for e in H.edges() if e not in edges_to_plant and (e[1], e[0]) not in edges_to_plant]
    to_remove = random.sample(removable, k)
    H.remove_edges_from(to_remove)

    # add the planted minor edges
    H.add_edges_from(edges_to_plant)

    # sanity check: now it's non‐planar
    assert not nx.check_planarity(H, False)[0]
    return H

# ── Worker functions ───────────────────────────────────────────────────────────
def make_and_save_planar(i):
    n = random.randint(*NODE_RANGE)
    G = random_sparse_planar(n)
    A = nx.to_numpy_array(G, dtype=np.int8)
    fn = f"planar_{i:04d}_n{n}.npy"
    np.save(os.path.join(PLANAR_DIR, fn), A)

def make_and_save_nonplanar(i):
    n = random.randint(*NODE_RANGE)
    G = random_sparse_planar(n)
    H = plant_forbidden_minor(G)
    A = nx.to_numpy_array(H, dtype=np.int8)
    fn = f"nonplanar_{i:04d}_n{n}.npy"
    np.save(os.path.join(NONPLANAR_DIR, fn), A)

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with Pool(cpu_count()) as pool:
        pool.map(make_and_save_planar,    range(NUM_GRAPHS))
        pool.map(make_and_save_nonplanar, range(NUM_GRAPHS))
    print(f"Finished: {NUM_GRAPHS} sparse planar and {NUM_GRAPHS} near-sparse non-planar graphs.")
