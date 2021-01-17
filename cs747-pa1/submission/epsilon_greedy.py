#! /usr/bin/env python3

import random
from bandit_algo import BanditAlgorithm


class EpsilonGreedy(BanditAlgorithm):
    def __init__(self, params):
        super().__init__(params)
        self.expected = [[0, 0]]*self.no_arms

    def update(self, arm):
        rwd = self.reward(arm)
        mean, run = self.expected[arm]
        new_mean, new_run = (mean*run + rwd)/(run+1), run+1
        self.expected[arm] = [new_mean, new_run]
        self.total_reward += rwd

    def initialize_arms(self):
        for a in range(self.no_arms):
            self.update(a)
        self.total_reward = 0

    def run(self):
        self.initialize_arms()
        for t in range(self.horizon):
            if (self.ep > random.random()):
                a = random.randint(0, self.no_arms-1)
            else:
                a = 0
                for i in range(self.no_arms):
                    if (self.expected[a][0] < self.expected[i][0]):
                        a = i
            self.update(a)
