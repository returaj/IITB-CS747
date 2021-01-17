#! /usr/bin/env python3

import math
import random
import numpy as np
from bandit_algo import BanditAlgorithm


class ThompsonSamplingWithHint(BanditAlgorithm):
    def __init__(self, params):
        super().__init__(params)
        self.hint = np.sort(self.arms)
        self.arms_dist = np.ones((self.no_arms, self.no_arms))
        self.expected = []

    def update_dist(self, a, rwd):
        self.arms_dist[a] *= [(x**rwd) * ((1-x)**(1-rwd)) for x in self.hint]
        if np.all(self.arms_dist[a] < 1e-4):
            self.arms_dist[a] *= 1e4

    def sample_dist(self, arm):
        prob = self.arms_dist[arm]/sum(self.arms_dist[arm])
        u = random.random()
        p = 0;
        for a in range(len(self.hint)):
            p += prob[a]
            if u < p:
                return self.hint[a]
        raise Exception("Cannot reach this position")

    def initialize_arms(self):
        for a in range(self.no_arms):
            rwd = self.reward(a)
            self.update_dist(a, rwd)
            self.expected.append([rwd, 1-rwd, self.sample_dist(a)])
        self.total_reward = 0

    def update(self, arm):
        rwd = self.reward(arm)
        for aid in range(self.no_arms):
            if aid == arm:
                self.expected[aid][0] += rwd
                self.expected[aid][1] += 1-rwd
                self.update_dist(aid, rwd)
            self.expected[aid][2] = self.sample_dist(aid)
        self.total_reward += rwd

    def run(self):
        self.initialize_arms()
        for t in range(self.horizon):
            a = 0
            for i in range(self.no_arms):
                if self.expected[a][2] < self.expected[i][2]:
                    a = i
            self.update(a)


