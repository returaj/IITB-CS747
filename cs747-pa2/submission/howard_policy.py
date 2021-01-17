#! /usr/bin/python3

import numpy as np
from mdp_solver import MDPSolver


class HowardPolicyIteration(MDPSolver):
    def __init__(self, mdp_path):
        super().__init__(mdp_path)

    def cal_v_pi(self, pi):
        A = []; B = []
        for s in range(self.num_states):
            if s in self.end:
                continue
            tmp = -self.g * self.T[s, :, pi[s]]
            tmp[s] += 1
            A.append(tmp)
            B.append(np.sum(self.T[s, :, pi[s]]*self.R[s, :, pi[s]]))
        a = np.array(A); b = np.array(B)
        if self.has_end:
            a = np.delete(a, self.end, 1) 
        v = np.linalg.solve(a, b)
        if not self.has_end:
            return v
        ret = []; p = 0;
        for x in v:
            while p in self.end:
                ret.append(0)
                p += 1
            ret.append(x)
            p += 1
        return ret

    def improve_action(self, pi):
        v = self.cal_v_pi(pi)
        is_improved = False
        for s in range(self.num_states):
            q = np.sum(self.T[s]*self.R[s], axis=0) + self.g*np.dot(v, self.T[s])
            a = np.argmax(q)
            if a != pi[s]:
                pi[s] = a
                is_improved = True
        return is_improved

    def run(self):
        pi = [0]*self.num_states
        is_improved = True
        while is_improved:
            is_improved = self.improve_action(pi)
        self.a_star = pi
        self.v_star = list(self.cal_v_pi(pi))


