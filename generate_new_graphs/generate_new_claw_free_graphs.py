import os
import random
import argparse
import numpy as np
import networkx as nx
from math import ceil

def generate_clawfree_adjacency(n, m_target, seed=None):
    """
    1) Build H = random_tree(n+1) ⇒ L = line_graph(H) has n nodes, (n-1) edges.
    2) Add up to (m_target - current) random edges to L, clamped to available non-edges.
    """
    if seed is not None:
        random.seed(seed)
    # base claw-free
    H = nx.random_tree(n+1, seed=seed)
    G = nx.line_graph(H)

    # clamp how many edges we can add
    cur = G.number_of_edges()
    non_edges = list(nx.non_edges(G))
    to_add = m_target - cur

    if to_add < 0:
        print(f"⚠️ Warning: base edges={cur} > target={m_target}. "
              "Skipping removal; density will be higher than requested.")
        to_add = 0
    elif to_add > len(non_edges):
        max_possible = cur + len(non_edges)
        print(f"⚠️ Warning: target={m_target} > max edges={max_possible}. "
              f"Clamping to {max_possible}.")
        to_add = len(non_edges)

    if to_add:
        extras = random.sample(non_edges, to_add)
        G.add_edges_from(extras)

    return G

def generate_nonclawfree_adjacency(n, m_target, claws, seed=None):
    """
    1) Start from a claw-free graph of size (n, m_target).
    2) Plant `claws` disjoint K1,3’s by adding 3 edges each.
    3) Remove 3*claws other edges at random to restore |E| = m_target.
    """
    if seed is not None:
        random.seed(seed)

    # base claw-free at correct edge-count
    G = generate_clawfree_adjacency(n, m_target, seed=seed)

    # use a separate random tree to pick disjoint edge sets
    H = nx.random_tree(n+1, seed=seed)
    H_edges = list(H.edges())

    planted = 0
    used = set()
    attempts = 0
    while planted < claws and attempts < claws * 50:
        attempts += 1
        e0, e1, e2, e3 = random.sample(H_edges, 4)
        # ensure disjointness in H ⇒ isolated in L(H)
        if (set(e0).isdisjoint(e1) and set(e0).isdisjoint(e2) and
            set(e0).isdisjoint(e3) and set(e1).isdisjoint(e2) and
            set(e1).isdisjoint(e3) and set(e2).isdisjoint(e3)):
            for leaf in (e1, e2, e3):
                G.add_edge(e0, leaf)
                used.add(tuple(sorted((e0, leaf))))
            planted += 1

    # remove other edges to restore the original count
    remove_count = 3 * planted
    removable = [e for e in G.edges() if tuple(sorted(e)) not in used]
    if remove_count > len(removable):
        raise RuntimeError(f"Not enough edges to remove: need {remove_count}, "
                           f"have {len(removable)}")
    to_remove = random.sample(removable, remove_count)
    G.remove_edges_from(to_remove)

    return G

def save_graphs(num, nmin, nmax, density, claws, cf_dir, ncf_dir, seed):
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(cf_dir,  exist_ok=True)
    os.makedirs(ncf_dir, exist_ok=True)

    for i in range(num):
        n = random.randint(nmin, nmax)
        m_target = int(density * (n * (n - 1) // 2))

        G_cf  = generate_clawfree_adjacency(n, m_target, seed=seed + i)
        G_ncf = generate_nonclawfree_adjacency(n, m_target, claws,
                                               seed=seed + num + i)

        A_cf  = nx.to_numpy_array(G_cf,  nodelist=sorted(G_cf.nodes()),  dtype=np.int8)
        A_ncf = nx.to_numpy_array(G_ncf, nodelist=sorted(G_ncf.nodes()), dtype=np.int8)

        np.save(os.path.join(cf_dir,   f"clawfree_{i:04d}.npy"),   A_cf)
        np.save(os.path.join(ncf_dir, f"nonclawfree_{i:04d}.npy"), A_ncf)

        if (i + 1) % 100 == 0:
            print(f"Generated {i+1}/{num}: n={n}, m={m_target}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate claw-free & non-claw-free graphs with controlled density"
    )
    parser.add_argument("--num",     type=int,   default=2000, help="Graphs per class")
    parser.add_argument("--n-min",   type=int,   default=401,  help="Minimum number of nodes")
    parser.add_argument("--n-max",   type=int,   default=500, help="Maximum number of nodes")
    parser.add_argument("--density", type=float, default=0.008, help="Target density ρ (0<ρ<1)")
    parser.add_argument("--claws",   type=int,   default=20,   help="K1,3’s to plant in non-claw-free")
    parser.add_argument("--cf-dir",  type=str,   default="clawfree_graphs",     help="Output directory for claw-free")
    parser.add_argument("--ncf-dir", type=str,   default="nonclawfree_graphs", help="Output directory for non-claw-free")
    parser.add_argument("--seed",    type=int,   default=42,    help="Random seed")
    args = parser.parse_args()

    save_graphs(
        num     = args.num,
        nmin    = args.n_min,
        nmax    = args.n_max,
        density = args.density,
        claws   = args.claws,
        cf_dir  = args.cf_dir,
        ncf_dir = args.ncf_dir,
        seed    = args.seed
    )
    print("✅ Done. Both classes at density", args.density)
