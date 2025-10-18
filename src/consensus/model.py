# src/consensus/model.py
import random
import numpy as np
import networkx as nx
from mesa import Model
from .agent import ConsensusAgent

class ConsensusModel(Model):
    """
    Consensus model supporting 'metropolis', 'simple_avg' and 'gossip' protocols.
    For gossip we perform a single random-edge averaging per step (asynchronous).
    """
    def __init__(self, N=50, graph_type='erdos_renyi', graph_params=None,
                 alpha=0.5, protocol='metropolis', noise_std=0.0,
                 p_drop=0.0, seed=None):
        super().__init__(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.num_agents = N
        self.alpha = alpha
        self.protocol = protocol
        self.noise_std = noise_std
        self.p_drop = p_drop

        # Graph creation
        if graph_params is None:
            graph_params = {}
        
        elif graph_type == 'erdos_renyi':
            p = graph_params.get('p', 0.1)
            self.G = nx.erdos_renyi_graph(N, p, seed=seed)
             # Ensure connectedness
            tries = 0
            while not nx.is_connected(self.G) and tries < 10:
                tries += 1
                self.G = nx.erdos_renyi_graph(N, p, seed=random.randint(0, 9999))



        elif graph_type == 'ring':
            self.G = nx.cycle_graph(N)
        elif graph_type == 'watts_strogatz':
            k = graph_params.get('k', 4)
            p = graph_params.get('p', 0.1)
            self.G = nx.watts_strogatz_graph(N, k, p, seed=seed)
        elif graph_type == 'barabasi_albert':
            m = graph_params.get('m', 2)
            self.G = nx.barabasi_albert_graph(N, m, seed=seed)
        else:
            raise ValueError("unknown graph_type")

        # create agents: instantiate ConsensusAgent registers them in model.agents
        self.node_agent_map = {}
        for node in self.G.nodes():
            init = random.uniform(0, 1)
            a = ConsensusAgent(self, node, init_value=init)
            self.node_agent_map[node] = a

        # precompute Metropolis weights if needed
        if self.protocol == 'metropolis':
            self.W = self.compute_metropolis_weights()

        # history
        self.history = []
        self.step_count = 0

    def compute_metropolis_weights(self):
        W = {i: {} for i in self.G.nodes()}
        for i in self.G.nodes():
            for j in self.G.neighbors(i):
                W[i][j] = 1.0 / (1 + max(self.G.degree[i], self.G.degree[j]))
            W[i][i] = 1.0 - sum(W[i].values())
        return W

    def try_drop_message(self):
        return random.random() < self.p_drop

    def byzantine_value(self, receiver_id, sender_id):
        # Example Byzantine strategy: fixed large bias
        return 10.0

    def gossip_step(self):
        """
        Perform a single pairwise gossip update:
        choose random edge (u,v) and set both endpoints to weighted average:
        x_u <- x_u + alpha*(x_v - x_u)
        x_v <- x_v + alpha*(x_u_old - x_v)
        We apply updates immediately (asynchronous).
        """
        if self.G.number_of_edges() == 0:
            return
        u, v = random.choice(list(self.G.edges()))
        # simulate message drop individually for each direction
        u_val = self.node_agent_map[u].value
        v_val = self.node_agent_map[v].value

        # Byzantine handling (they report manipulated value)
        if self.node_agent_map[u].is_byzantine:
            reported_u = self.byzantine_value(v, u)
        else:
            reported_u = u_val
        if self.node_agent_map[v].is_byzantine:
            reported_v = self.byzantine_value(u, v)
        else:
            reported_v = v_val

        # potential packet drops
        if self.try_drop_message():
            # drop message from v->u: u sees nothing => u keeps its value
            new_u = u_val
        else:
            new_u = u_val + self.alpha * (reported_v - u_val)
            if self.noise_std > 0:
                new_u += np.random.normal(0, self.noise_std)

        if self.try_drop_message():
            new_v = v_val
        else:
            new_v = v_val + self.alpha * (reported_u - v_val)
            if self.noise_std > 0:
                new_v += np.random.normal(0, self.noise_std)

        # commit immediately
        self.node_agent_map[u].value = new_u
        self.node_agent_map[v].value = new_v

    def step(self):
        """
        One simulation step. For synchronous protocols we call agent.step()/advance();
        for gossip we call gossip_step (asynchronous).
        """
        if self.protocol == 'gossip':
            # gossip step - single edge update
            self.gossip_step()
            # collect stats
            self.step_count += 1
        else:
            # synchronous: compute next values then commit
            self.agents.do("step")
            self.agents.do("advance")
            self.step_count += 1

        # collect stats
        vals = np.array([a.value for a in self.node_agent_map.values()])  # <-- convert to np.array
        mean = float(np.mean(vals))
        var = float(np.var(vals))
        rng = float(np.max(vals) - np.min(vals))
        l2_error = float(np.sum((vals - mean) ** 2))  # now this works fine

        self.history.append({
            'step': self.step_count,
            'mean': mean,
            'var': var,
            'range': rng,
            'l2_error': l2_error
        })

    def run_until(self, max_steps=1000, tol_range=1e-4):
        for _ in range(max_steps):
            self.step()
            if self.history and self.history[-1]['range'] < tol_range:
                break
        return self
