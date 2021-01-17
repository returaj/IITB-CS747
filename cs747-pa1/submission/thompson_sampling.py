#! /usr/bin/env python3

import math
import numpy as np
from bandit_algo import BanditAlgorithm


class ThompsonSampling(BanditAlgorithm):
    def __init__(self, params):
        super().__init__(params)
        np.random.seed(params.randomSeed)
        self.expected = []

    def initialize_arms(self):
        for a in range(self.no_arms):
            rwd = self.reward(a)
            self.expected.append([rwd, 1-rwd, np.random.beta(rwd+1, 2-rwd)])
        self.total_reward = 0

    def update(self, arm):
        rwd = self.reward(arm)
        for aid in range(self.no_arms):
            if aid == arm:
                self.expected[aid][0] += rwd
                self.expected[aid][1] += 1-rwd
            a, b, x = self.expected[aid]
            self.expected[aid][2] = np.random.beta(a+1, b+1)
        self.total_reward += rwd

    def run(self):
        self.initialize_arms()
        for t in range(self.horizon):
            a = 0
            for i in range(self.no_arms):
                if self.expected[a][2] < self.expected[i][2]:
                    a = i
            self.update(a)

