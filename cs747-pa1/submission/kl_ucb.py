#! /usr/bin/env python3

import math
from bandit_algo import BanditAlgorithm


class KL_UCB(BanditAlgorithm):
    def __init__(self, params):
        super().__init__(params)
        self.expected = []

    def kl(self, p, q):
        ft = 0.0 if p == 0 else p*math.log(p/q)
        st = 0.0 if p == 1 else (1-p)*math.log((1-p)/(1-q))
        return ft+st

    def initialize_arms(self):
        for a in range(self.no_arms):
            rwd = self.reward(a)
            self.expected.append([rwd, 1, rwd])
        self.total_reward = 0

    def update_qmax(self, arm, time, c=3, ep=0.001):
        p, u, _ = self.expected[arm]
        val = math.log(time) + c*math.log(math.log(1+time))
        l = p; r = 1
        while (l < r and r-l > ep):
            mid = l+(r-l)/2
            if (u*self.kl(p, mid) > val):
                r = mid
            else:
                l = mid
        self.expected[arm][2] = l

    def update(self, arm, time):
        rwd = self.reward(arm)
        for a in range(self.no_arms):
            if a == arm:
                mean, run, _ = self.expected[a]
                self.expected[a][0] = (mean*run + rwd)/(run+1)
                self.expected[a][1] = run+1
            self.update_qmax(a, time)
        self.total_reward += rwd

    def run(self):
        self.initialize_arms()
        for t in range(self.horizon):
            a = 0
            for i in range(self.no_arms):
                if self.expected[a][2] < self.expected[i][2]:
                    a = i
            self.update(a, t+1)

