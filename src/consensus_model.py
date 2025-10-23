# pip install mesa networkx numpy pandas matplotlib
import random, math
import numpy as np
import networkx as nx
from mesa import Agent, Model

import matplotlib.pyplot as plt
import pandas as pd


class ConsensusAgent(Agent):

    def __init__(self, model, node_id, init_value=0.0):
        super().__init__(model)
        self.node_id = node_id       
        self.value = float(init_value)
        self.next_value = float(init_value)
        self.is_stubborn = False
        self.is_byzantine = False

    def step(self):
        if self.is_stubborn:
            self.next_value = self.value
            return

        nbrs = list(self.model.G.neighbors(self.node_id))
        if len(nbrs) == 0:
            self.next_value = self.value
            return

        if self.model.protocol == 'simple_avg':
            observed = []
            for j in nbrs:
                if self.model.try_drop_message():
                    continue
                v = self.model.node_agent_map[j].value
                if self.model.node_agent_map[j].is_byzantine:
                    v = self.model.byzantine_value(self.node_id, j)
                observed.append(v)
            if len(observed) == 0:
                self.next_value = self.value
            else:
                avg = sum(observed) / len(observed)
                self.next_value = self.value + self.model.alpha * (avg - self.value)

        elif self.model.protocol == 'metropolis':
            row = self.model.W[self.node_id]
            s = 0.0
            for j, w in row.items():
                if j == self.node_id:
                    s += w * self.value
                else:
                    if self.model.try_drop_message():
                        continue
                    v = self.model.node_agent_map[j].value
                    if self.model.node_agent_map[j].is_byzantine:
                        v = self.model.byzantine_value(self.node_id, j)
                    s += w * v
            self.next_value = s

        if self.model.noise_std > 0:
            self.next_value += np.random.normal(0, self.model.noise_std)

    def advance(self):
        self.value = self.next_value


class ConsensusModel(Model):
    """
    Consensus model adapted to Mesa 3.x APIs.
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

        if graph_params is None:
            graph_params = {}

        if graph_type == 'erdos_renyi':
            p = graph_params.get('p', 0.1)
            self.G = nx.erdos_renyi_graph(N, p, seed=seed)
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

        self.node_agent_map = {}
        for node in self.G.nodes():
            init = random.uniform(0, 1)
            a = ConsensusAgent(self, node, init_value=init)
            self.node_agent_map[node] = a

        if self.protocol == 'metropolis':
            self.W = self.compute_metropolis_weights()

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
        return 10.0

    def step(self):
        """
        Perform a simultaneous step:
          1) call all agents' step() (compute next_value)
          2) call all agents' advance() (commit next_value)
        Then collect statistics.
        """
       
        self.agents.do("step")
        # then commit/advance
        self.agents.do("advance")

        self.step_count += 1
        vals = [a.value for a in self.node_agent_map.values()]
        mean = float(np.mean(vals))
        var = float(np.var(vals))
        rng = float(max(vals) - min(vals))
        self.history.append({'step': self.step_count, 'mean': mean, 'var': var, 'range': rng})

    def run_until(self, max_steps=1000, tol_range=1e-4):
        for _ in range(max_steps):
            self.step()
            if self.history and self.history[-1]['range'] < tol_range:
                break
        return self


if __name__ == "__main__":
    m = ConsensusModel(N=50, graph_type='erdos_renyi', graph_params={'p': 0.08},
                       alpha=0.5, protocol='metropolis', noise_std=0.0,
                       p_drop=0.0, seed=42)
    m.run_until(max_steps=500, tol_range=1e-5)

    df = pd.DataFrame(m.history)
    print(df.tail())

    fig, ax1 = plt.subplots()
    ax1.plot(df['step'], df['mean'], label='mean')
    ax1.set_xlabel('step')
    ax1.set_ylabel('mean')
    ax2 = ax1.twinx()
    ax2.plot(df['step'], df['range'], label='range', linestyle='--')
    ax2.set_ylabel('range')
    plt.title('Consensus: mean (left) and range (right)')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.show()
