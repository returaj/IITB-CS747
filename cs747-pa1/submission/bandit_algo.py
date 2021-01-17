#! /usr/bin/env python3

import random


class BanditAlgorithm:
    def __init__(self, params):
        random.seed(params.randomSeed)
        self.instance = params.instance
        self.algo = params.algorithm
        self.seed = params.randomSeed
        self.arms = self.read_bandit_instance(params.instance)
        self.no_arms = len(self.arms)
        self.ep = params.epsilon
        self.horizon = params.horizon
        self.total_reward = 0

    def read_bandit_instance(self, path):
        arms = []
        with open(path, 'r') as fp:
            for line in fp:
                arms.append(float(line.strip()))
        return arms

    def print_rewards(self):
        print(f"Algorithm: {self.__class__.__name__}")
        print(f"ExpectedReward: {self.total_reward}")
        print(f"MaxReward: {self.max_reward()}")
        print(f"MinReward: {self.min_reward()}")
        print(f"AvgReward: {self.avg_reward()}")

    def save_output(self, filepath):
        reg = self.max_reward() - self.total_reward
        with open(filepath, 'a') as fp:
            fp.write(f"{self.instance}, {self.algo}, {self.seed}, {self.ep}, {self.horizon}, {reg}\n")

    def print_output(self):
        reg = self.max_reward() - self.total_reward
        print(f"{self.instance}, {self.algo}, {self.seed}, {self.ep}, {self.horizon}, {reg}")

    def max_reward(self):
        p = max(self.arms)
        return p*self.horizon

    def min_reward(self):
        p = min(self.arms)
        return p*self.horizon

    def avg_reward(self, h):
        p = sum(self.arms)/self.no_arms
        return p*self.horizon

    def reward(self, arm):
        if (self.arms[arm] > random.random()):
            return 1
        return 0

    def initialize_arms(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

