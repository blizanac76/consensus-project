# src/consensus/agent.py
from mesa import Agent
import numpy as np

class ConsensusAgent(Agent):
    """
    node_id - id cvora
    value - vrednost koju on cuva, njegovo merenje
    next_value - nova vrednost u sledecem koraku. posle commita ovo postaje value
    is_stubborn - tvrdoglavost (true - ne menja svoju vrednost, ne veruje nikom)
    is_byzantine - da li je agent vizantijski (salje lose vrednosti, ekstremne u ovom slucaju)
    """
    def __init__(self, model, node_id, init_value=0.0):
        super().__init__(model)
        self.node_id = node_id
        self.value = float(init_value)
        self.next_value = float(init_value)
        self.is_stubborn = False
        self.is_byzantine = False

    def step(self):
        """
        step se poziva u svakoj iteraciji kod svakog agenta.

        """
        # ako je stubborn zadrzava svoju vrednost
        if self.is_stubborn:
            self.next_value = self.value
            return
        # susedi u grafu
        nbrs = list(self.model.G.neighbors(self.node_id))
        if len(nbrs) == 0:
            self.next_value = self.value
            return
        # ako je protokol simple_avg
        if self.model.protocol == 'simple_avg':
            observed = []
            for j in nbrs:
                # gubitak poruke, nastavlja se dalje
                if self.model.try_drop_message():
                    continue
                v = self.model.node_agent_map[j].value
                # ako je sused byzantine, koristi se lazna vrednost
                if self.model.node_agent_map[j].is_byzantine:
                    v = self.model.byzantine_value(self.node_id, j)
                observed.append(v)
            #ako nije primljena ni jedna poruka od suseda, zadrzi staru vrednost
            if len(observed) == 0:
                self.next_value = self.value
            else:
                # u suprotnom racunaj novu vrednost sa korakom alfa i prosekom. simple avg- aritmetika
                avg = sum(observed) / len(observed)
                self.next_value = self.value + self.model.alpha * (avg - self.value)

        #metropolis protokol
        elif self.model.protocol == 'metropolis':
            # uzimaju se unapred izracunate tezine iz matrice W
            row = self.model.W[self.node_id]
            #s akumulator, na kraju je to next value
            s = 0.0
            #j agent,, w tezina agenta. iterira se kroz sve cvorove j (susedi i sam agent), i za svakog se uzima njihova trenutna vrednost (v) - ona se ponderise tezinom w
            #ako je poruka dropovana, vrednost se ignorise! ako je vizantijski agent, uzima se lazna vrednost (ovde je to 10 neki extrem kao)
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

        # dodavanje suma ako je noise_std > 0
        if self.model.noise_std > 0:
            self.next_value += np.random.normal(0, self.model.noise_std)

    def advance(self):
        # commit novih izmena na value
        self.value = self.next_value
