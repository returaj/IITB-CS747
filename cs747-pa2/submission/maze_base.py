#! /usr/bin/python3


class BaseMaze:
    def __init__(self, gridfile):
        self.num_actions = 4
        self.move = ["N", "S", "W", "E"]
        self.xmove = [-1, 1, 0, 0]
        self.ymove = [0, 0, -1, 1]
        self.grid = self.load_grid(gridfile)
        self.state2id = {}
        self.id2state = {}
        self.num_states = 0
        self.start = 0
        self.end = []
        self.map_all_states()

    def load_grid(self, gridfile):
        g = []
        with open(gridfile, 'r') as fp:
            for line in fp:
                g.append(line.strip().split(" "))
        return g

    def map_all_states(self):
        m = len(self.grid); n = len(self.grid[0])
        for s in range(m*n):
            x = s//n; y = s%n;
            if self.grid[x][y] == "1":
                continue
            if self.grid[x][y] == "2":
                self.start = self.num_states
            elif self.grid[x][y] == "3":
                self.end.append(self.num_states)
            self.state2id[s] = self.num_states
            self.id2state[self.num_states] = s
            self.num_states += 1

