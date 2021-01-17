#! /usr/bin/python3

import argparse
from maze_base import BaseMaze


class DecodeMaze(BaseMaze):
    def __init__(self, gridfile, policyfile):
        super().__init__(gridfile)
        self.policy = self.load_policy(policyfile)
        self.path = []

    def load_policy(self, policyfile):
        p = []
        with open(policyfile, 'r') as fp:
            for line in fp:
                p.append(line.strip().split(" "))
        return p

    def decode(self):
        m = len(self.grid); n = len(self.grid[0])
        s = self.start
        while s not in self.end:
            if int(float(self.policy[s][0])) <= -m*n:
                raise Exception("Cannot reach from this state %d, %d", (s//n, s%n))
            a = int(self.policy[s][1])
            self.path.append(self.move[a])
            a_s = self.id2state[s]
            sn = a_s//n + self.xmove[a]; sy = a_s%n + self.ymove[a]
            s = self.state2id[sn*n + sy]

    def print_path(self):
        print(" ".join(self.path))

    def print_policy(self):
        m = len(self.grid); n = len(self.grid[0])
        for i in range(m*n):
            if i % n == 0:
                print()
            if self.grid[i//n][i%n] == "3":
                print("X", end=" ")
            elif self.grid[i//n][i%n] == "1":
                print("|", end=" ")
            else:
               print("%4.1f" % float(self.policy[self.state2id[i]][0]), end=" ")
        print()

    def print_action(self):
        m = len(self.grid); n = len(self.grid[0])
        for i in range(m*n):
            if i % n == 0:
                print()
            if self.grid[i//n][i%n] == "3":
                print("X", end=" ")
            elif self.grid[i//n][i%n] == "1":
                print("|", end=" ")
            else:
                a = int(self.policy[self.state2id[i]][1])
                print(self.move[a], end=" ")
        print()


def main(params):
    maze = DecodeMaze(params.grid, params.value_policy)
    maze.decode()
    maze.print_path()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--grid", help="grid file path", required=True)
    p.add_argument("--value_policy", help="policy file", required=True)
    main(p.parse_args())




