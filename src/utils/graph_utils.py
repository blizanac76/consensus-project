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
        # small-world graf: k je broj komsija, p je verovatnoca rewiring-a odnosno random vezama sa nekim drugim cvorovima umesto suseda
        k = graph_params.get('k', 4)
        p = graph_params.get('p', 0.2)
        return nx.watts_strogatz_graph(N, k, p, seed=seed)
    elif graph_type == 'barabasi_albert':
        # scale-free graf: m je broj novih veza koje novi cvor pravi
        m = graph_params.get('m', 2)
        return nx.barabasi_albert_graph(N, m, seed=seed)
    else:
        raise ValueError("unknown graph_type")

def spectral_gap(G):
    """
    racuna spektralni gap laplasijana grafa (lampda_2)
    ako graf nije povezan, lampda_2 ce biti 0
    """

    L = nx.laplacian_matrix(G).toarray()
    eigs = np.linalg.eigvals(L)
    eigs_sorted = np.sort(np.real(eigs))
    # ako ih ima manje od 2, vraca 0
    if len(eigs_sorted) < 2:
        return 0.0
    # inace vraca drugu najmanju vrednost = lambda_2
    return float(eigs_sorted[1])
