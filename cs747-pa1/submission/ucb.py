#! /usr/bin/env python3

import math
from bandit_algo import BanditAlgorithm


class UCB(BanditAlgorithm):
    def __init__(self, params):
        super().__init__(params)
        self.expected = []

    def expected_run(self):
        pmax = max(self.arms)
        run = 0
        for a in range(self.no_arms):
            if pmax == self.arms[a]:
                continue
            run += int(8*math.log(self.horizon)/(pmax-self.arms[a]))
        return run

    def initialize_arms(self):
        for a in range(self.no_arms):
            rwd = self.reward(a)
            self.expected.append([rwd, 1, rwd])
        self.total_reward = 0

    def update(self, arm, time):
        rwd = self.reward(arm)
        for a in range(self.no_arms):
            if a == arm:
                mean, run, _ = self.expected[a]
                self.expected[a][0] = (mean*run + rwd)/(run+1)
                self.expected[a][1] = run+1
            mean, run, _ = self.expected[a]
            self.expected[a][2] = mean + math.sqrt(2*math.log(time)/run)
        self.total_reward += rwd

    def run(self):
        self.initialize_arms()
        for t in range(self.horizon):
            a = 0
            for i in range(self.no_arms):
                if self.expected[a][2] < self.expected[i][2]:
                    a = i
            self.update(a, t+1)

