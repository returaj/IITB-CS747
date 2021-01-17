#! /usr/bin/python3

import argparse
import numpy as np
from value_iteration import ValueIteration
from howard_policy import HowardPolicyIteration
from linear_programming import LinearProgramming
from mdp_solver import MDPSolver


MAP_ALGORITHM = {
    "vi": ValueIteration,
    "hpi": HowardPolicyIteration,
    "lp": LinearProgramming
}


def main(params):
    solver = MAP_ALGORITHM[params.algorithm](params.mdp)
    solver.run()
    solver.print()


def get_args(argv=None):
    parser = argparse.ArgumentParser(description="solve mdp planning problem")
    parser.add_argument("--mdp", help="mdp filepath", required=True)
    parser.add_argument("--algorithm", help="alogirthm name", required=True)
    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':
    main(get_args())

    
