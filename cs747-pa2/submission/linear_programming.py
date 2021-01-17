#! /usr/bin/python3

import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, LpStatusOptimal, lpSum, PULP_CBC_CMD
from mdp_solver import MDPSolver


class LinearProgramming(MDPSolver):
    def __init__(self, mdp_path):
        super().__init__(mdp_path)

    def add_constraints(self, v, prob):
        for s in range(self.num_states):
            if s in self.end:
                continue
            c = np.sum(self.T[s] * self.R[s], axis=0)
            t_ = -self.g*self.T[s]
            t_[s] += 1
            if self.has_end:
                t_ = np.delete(t_, self.end, 0)
            f = np.dot(v, t_)
            for a in range(self.num_actions):
                prob += f[a]>=c[a], f"V{s}_A{a}_constraints"

    def get_action(self, V):
        A = []
        for s in range(self.num_states):
            q = np.sum(self.T[s]*self.R[s], axis=0) + self.g*np.dot(V, self.T[s])
            A.append(np.argmax(q)) 
        return A

    def run(self):
        prob = LpProblem("V_star_Problem", LpMinimize)
        
        var = []; state_to_var = {}
        for s in range(self.num_states):
            if s not in self.end:
                state_to_var[f"V{s}"] = s
                var.append(LpVariable(f"V{s}", cat="Continuous"))

        prob += lpSum(var), "minimize_V"
        self.add_constraints(var, prob)

        prob.writeLP("CalculateVStarProblem.lp")
        result = prob.solve(PULP_CBC_CMD(msg=0))
        assert LpStatusOptimal == result

        self.v_star = [0]*self.num_states
        for v in prob.variables():
            if v.name in state_to_var:
                self.v_star[state_to_var[v.name]] = v.varValue

        self.a_star = self.get_action(self.v_star)
 
