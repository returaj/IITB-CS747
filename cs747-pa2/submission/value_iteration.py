#! /usr/bin/python3

import numpy as np
from mdp_solver import MDPSolver


class ValueIteration(MDPSolver):
    def __init__(self, mdp_file, ep=1e-8):
        super().__init__(mdp_file)
        self.ep = ep
        self.max_run = int(1e8)

    def bellman_opti(self, s, V):
        if s in self.end:
            return 0 
        return np.max(np.sum(self.T[s]*self.R[s], axis=0) + self.g*np.dot(V, self.T[s]))

    def get_action(self, V):
        A = []
        for s in range(self.num_states):
            q = np.sum(self.T[s]*self.R[s], axis=0) + self.g*np.dot(V, self.T[s])
            A.append(np.argmax(q)) 
        return A

    def run(self):
        Vnew = 1/self.num_states * np.ones(self.num_states)
        if self.end[0] != -1:
            Vnew[self.end] = 0
        Vprev = np.zeros(self.num_states)
        is_continue = True
        run = 0
        while is_continue and run < self.max_run:
            run += 1
            is_continue = False
            tmp = Vnew
            Vnew = Vprev
            Vprev = tmp
            for s in range(self.num_states):
                Vnew[s] = self.bellman_opti(s, Vprev)
                if abs(Vnew[s] - Vprev[s]) > self.ep:
                    is_continue = True
        self.a_star = self.get_action(Vnew)
        self.v_star = list(Vnew)

