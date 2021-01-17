#!/usr/bin/python3

import numpy as np


class MDPSolver:
    def __init__(self, mdp_file):
        self.load_mdp(mdp_file)
        self.v_star = None
        self.a_star = None
        self.has_end = self.end[0] != -1

    def load_mdp(self, filepath):
        self.T = None; self.R = None;
        with open(filepath, 'r') as fp:
            for line in fp:
                arr = line.strip().split()
                if arr[0] == "numStates":
                    self.num_states = int(arr[1])
                elif arr[0] == "numActions":
                    self.num_actions = int(arr[1])
                elif arr[0] == "start":
                    self.start = int(arr[1])
                elif arr[0] == "end":
                    self.end = [int(x) for x in arr[1:]]
                elif arr[0] == "mdptype":
                    self.mdptype = arr[1]
                elif arr[0] == "discount":
                    self.g = float(arr[1])
                elif arr[0] == "transition":
                    if (self.T is None) or (self.R is None):
                        self.T = np.zeros((self.num_states, self.num_states, self.num_actions))
                        self.R = np.zeros((self.num_states, self.num_states, self.num_actions))
                    s1, a, s2, r, p = int(arr[1]), int(arr[2]), int(arr[3]), float(arr[4]), float(arr[5])
                    self.T[s1][s2][a] = p
                    self.R[s1][s2][a] = r
                else:
                    raise Exception("Invalid file format")

    def print(self):
        for s in range(self.num_states):
            print("{:.6f} {}".format(self.v_star[s], self.a_star[s]))

    def run(self):
        raise NotImplementedError("method not implemented")

