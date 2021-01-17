#! /usr/env/bin python

import argparse
from epsilon_greedy import EpsilonGreedy
from ucb import UCB
from kl_ucb import KL_UCB
from thompson_sampling import ThompsonSampling
from thompson_sampling_with_hint import ThompsonSamplingWithHint


MAP_ALGO = {
    "epsilon-greedy": EpsilonGreedy,
    "ucb": UCB,
    "kl-ucb": KL_UCB,
    "thompson-sampling": ThompsonSampling,
    "thompson-sampling-with-hint": ThompsonSamplingWithHint
}


def main(params):
    bandit = MAP_ALGO[params.algorithm](params)
    bandit.run()
    if params.output:
        bandit.save_output(params.output)
    else:
        bandit.print_output()        
#        bandit.print_rewards()

def get_args(argv=None):
    parser = argparse.ArgumentParser(description="read bandit instance")
    parser.add_argument("--instance", help="instance name", required=True)
    parser.add_argument("--algorithm", help="algorithm name", required=True)
    parser.add_argument("--randomSeed", help="random seed", required=True, type=int)
    parser.add_argument("--epsilon", help="epsilon for epsilon greedy algorithm", required=True, type=float)
    parser.add_argument("--horizon", help="length of the bandit instance", required=True, type=int)
    parser.add_argument("--output", help="output file path", required=False)
    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':
    main(get_args())
    
