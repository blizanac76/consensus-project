# src/consensus/model.py
import random
import numpy as np
import networkx as nx
from mesa import Model
from .agent import ConsensusAgent

class ConsensusModel(Model):
    """
    glavni model za simulaciju konsenzusa.
    podrzava protokole metropolis simple_avg gossip
    Gossip koristi asinhroni update preko random  edge
    """
    def __init__(self, N=50, graph_type='erdos_renyi', graph_params=None,
                 alpha=0.5, protocol='metropolis', noise_std=0.0,
                 p_drop=0.0, seed=None):
        super().__init__(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # parametri za simulaciju
        self.num_agents = N
        self.alpha = alpha # korak u update formulama
        self.protocol = protocol
        self.noise_std = noise_std
        self.p_drop = p_drop # VRV DA SE PORUKA ZAGUBI

        if graph_params is None:
            graph_params = {}
        
        elif graph_type == 'erdos_renyi':
            ## e-r graf G(n, p) --- nasumican graf sa n cvorova, svaka ivica/veza postoji sa verovatnocom p
            p = graph_params.get('p', 0.1)
            self.G = nx.erdos_renyi_graph(N, p, seed=seed)
             # osiguraj da je graf povezan (max 10 pokusaja)
            tries = 0
            while not nx.is_connected(self.G) and tries < 10:
                tries += 1
                self.G = nx.erdos_renyi_graph(N, p, seed=random.randint(0, 9999))



        elif graph_type == 'ring':
            ## ring graf -- svaki cvor je povezan sa 2 suseda (levo i desno), formira ciklus
            self.G = nx.cycle_graph(N)

        elif graph_type == 'watts_strogatz':
            # wtts–strogatz -- small-world graf
            # pocetna topologija je ring sa k suseda; zatim se nasumično rewira svaka ivica sa verovatnoćom p
            k = graph_params.get('k', 4)
            p = graph_params.get('p', 0.1)
            self.G = nx.watts_strogatz_graph(N, k, p, seed=seed)

        elif graph_type == 'barabasi_albert':
            # barabasialbert - scale-free mreza sa preferencijalnim povezivanjem
            m = graph_params.get('m', 2) # broj ivica koje svaki novi cvor dodaje
            self.G = nx.barabasi_albert_graph(N, m, seed=seed)
        else:
            raise ValueError("unknown graph_type")

        #  ---------- KREIRANJE AGENATA ZA SVAKI CVOR GRAFA ------------ #
        self.node_agent_map = {}
        for node in self.G.nodes():
            init = random.uniform(0, 1) # inicijalna vrednost iz uniformne raspodele 0 - 1
            a = ConsensusAgent(self, node, init_value=init) # svaki agent zna svoj node_id
            self.node_agent_map[node] = a

        # ako se koristi Metropolis protokol, unapred se racuna matrica tezina W
        if self.protocol == 'metropolis':
            self.W = self.compute_metropolis_weights()

        # istorija kao niz za cuvanje statistike
        self.history = []
        self.step_count = 0

    def compute_metropolis_weights(self):
        
        """
        racuna metropolis matricu tezina W:
        w_ij = 1 / (1 + max(d_i, d_j)) za susede i i   j
        w_ii = 1 - suma svih tezina ka susedima (da red sumira na 1)
        """

        W = {i: {} for i in self.G.nodes()}
        for i in self.G.nodes():
            for j in self.G.neighbors(i):
                W[i][j] = 1.0 / (1 + max(self.G.degree[i], self.G.degree[j]))
            W[i][i] = 1.0 - sum(W[i].values())
        return W

    def try_drop_message(self):
        return random.random() < self.p_drop

    def byzantine_value(self, receiver_id, sender_id):
        # neki veliki extrem, zarad simulacije moze 10
        return 10.0

    def gossip_step(self):
        """
        jedan asinhroni korak gossip protokola:
        - bira se nasumicna ivica (u,v)
        - oba agenta u i v update svoju vrednost kao u formuli:
          x_i postaje x_i + alfa(x_j - x_i)
        - moguca byzantine laznjak i gubitak poruke
        """
        if self.G.number_of_edges() == 0:
            return
        u, v = random.choice(list(self.G.edges()))
        # simulacija dropanja poruke u oba smera
        u_val = self.node_agent_map[u].value
        v_val = self.node_agent_map[v].value

         # bizantijski agenti šalju lažne vrednosti
        if self.node_agent_map[u].is_byzantine:
            reported_u = self.byzantine_value(v, u)
        else:
            reported_u = u_val
        if self.node_agent_map[v].is_byzantine:
            reported_v = self.byzantine_value(u, v)
        else:
            reported_v = v_val

        # agent u update vrednost ako poruka nije izgubljena
        if self.try_drop_message():
            new_u = u_val
        else:
            new_u = u_val + self.alpha * (reported_v - u_val)
            if self.noise_std > 0:
                new_u += np.random.normal(0, self.noise_std)
        # isto za agenta v
        if self.try_drop_message():
            new_v = v_val
        else:
            new_v = v_val + self.alpha * (reported_u - v_val)
            if self.noise_std > 0:
                new_v += np.random.normal(0, self.noise_std)

        # commit asinhroni, odmah
        self.node_agent_map[u].value = new_u
        self.node_agent_map[v].value = new_v

    def step(self):
        """
        jedan korak simulacije.
        - ako je gossip: pokrece jedan edge update
        - inace: svi agenti sinhrono racunaju (step) pa commituju (advance)
        - belezi se statistika stanja nakon svakog koraka
        """
        if self.protocol == 'gossip':
            # gossip step - single edge update
            self.gossip_step()
            self.step_count += 1
        else:
            # sihnroni: izracunaj sledecu vbrednost pa commituj
            self.agents.do("step")
            self.agents.do("advance")
            self.step_count += 1

        # skupljanje podataka
        vals = np.array([a.value for a in self.node_agent_map.values()])  # <-- convert to np.array
        mean = float(np.mean(vals)) #prosecna vr
        var = float(np.var(vals))   #varijansa
        rng = float(np.max(vals) - np.min(vals))    #range 
        l2_error = float(np.sum((vals - mean) ** 2))  # l2 suma kvadrata odstupanja

        self.history.append({
            'step': self.step_count,
            'mean': mean,
            'var': var,
            'range': rng,
            'l2_error': l2_error
        })

    def run_until(self, max_steps=1000, tol_range=1e-4):
        """
        pokrece model do konvergencije do tol_range vrednostti
        ili do max_steps koraka. Vraca model i podatke
        """
        for _ in range(max_steps):
            self.step()
            if self.history and self.history[-1]['range'] < tol_range:
                break
        return self
