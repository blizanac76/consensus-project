# src/utils/graph_utils.py
import numpy as np
import networkx as nx

def create_graph(graph_type, N, graph_params=None, seed=None):
    if graph_params is None:
        graph_params = {}
    if graph_type == 'erdos_renyi':
        p = graph_params.get('p', 0.1)
        return nx.erdos_renyi_graph(N, p, seed=seed)
    elif graph_type == 'ring':
        return nx.cycle_graph(N)
    elif graph_type == 'watts_strogatz':
        k = graph_params.get('k', 4)
        p = graph_params.get('p', 0.1)
        return nx.watts_strogatz_graph(N, k, p, seed=seed)
    elif graph_type == 'barabasi_albert':
        m = graph_params.get('m', 2)
        return nx.barabasi_albert_graph(N, m, seed=seed)
    else:
        raise ValueError("unknown graph_type")

def spectral_gap(G):
    """
    Compute Laplacian spectral gap lambda2 (second smallest eigenvalue).
    Returns float lambda2. If graph not connected, lambda2 will be 0.
    """
    L = nx.laplacian_matrix(G).toarray()
    eigs = np.linalg.eigvals(L)
    eigs_sorted = np.sort(np.real(eigs))
    if len(eigs_sorted) < 2:
        return 0.0
    return float(eigs_sorted[1])
