#! /usr/bin/python3

import argparse
import numpy as np
from maze_base import BaseMaze


class EncoderMaze(BaseMaze):
    def __init__(self, gridfile):
        super().__init__(gridfile)
        self.trans = []
        self.mdptype = "episodic"
        self.gama = 1.0

    def print(self):
        print("numStates", self.num_states)
        print("numActions", self.num_actions)
        print("start", self.start)
        print("end " + " ".join(map(str, self.end)))
        for t in self.trans:
            print(t)
        print("mdptype", self.mdptype)
        print("discount", self.gama)

    def check_all_dir(self, x, y):
        m = len(self.grid); n = len(self.grid[0])
        s = self.state2id[x*n + y]
        for a in range(self.num_actions):
            xn = x+self.xmove[a]; yn = y+self.ymove[a]
            sn = xn*n + yn
            if xn<0 or yn<0 or xn>=m or yn>=n or self.grid[xn][yn]=="1":
                self.trans.append(f"transition {s} {a} {s} -1 1")
            elif self.grid[xn][yn] == "3":
                self.trans.append(f"transition {s} {a} {self.state2id[sn]} 0 1")
            else:
                self.trans.append(f"transition {s} {a} {self.state2id[sn]} -1 1")

    def encode(self):
        m = len(self.grid); n = len(self.grid[0])
        for i in range(m):
            for j in range(n):
                if self.grid[i][j] == "1" or self.grid[i][j] == "3":
                    continue
                self.check_all_dir(i, j)


def main(param):
    maze = EncoderMaze(param.grid)
    maze.encode()
    maze.print()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Encode maze")
    p.add_argument("--grid", help="grid file", required=True)
    main(p.parse_args())

