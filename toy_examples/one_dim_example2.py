import _init_paths
from copy import deepcopy

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from utils.widgets import get_spare_dir


class Tester:

    def __init__(self, writer, gamma, T=100032):
        self.functions = [Tester.__quad_func(target) for target in [0, 2, 50, 100, -3]]
        self.target_it = 10
        self.its = [3, 20, 5, 10, 8]
        self.gamma = gamma
        self.T = T
        self.writer = writer

    @staticmethod
    def __quad_func(e):
        return {'ori': lambda x: (x - e) ** 2, 'grad': lambda x: 2 * (x - e)}

    def fedavg(self, x0, scaled=True):
        x = x0
        for t in range(self.T):
            next_x = deepcopy(x)
            gamma = self.gamma(t)
            for function, num_it in zip(self.functions, self.its):
                x_local = deepcopy(x)
                for it in range(num_it):
                    x_local -= gamma * function['grad'](x_local)
                x_update = x_local - x
                if scaled:
                    next_x += x_update * self.target_it / num_it
                else:
                    next_x += x_update
            x = next_x
            self.writer.add_scalar('scaled' if scaled else 'unscaled', x, t)
            print(f"{t}: {x}", end='\r')
        return x


if __name__ == '__main__':
    f_dir = 'runs'
    c_dir = 'test'
    log_dir = get_spare_dir(f_dir, c_dir)
    with SummaryWriter(log_dir) as writer:
        tester = Tester(writer, gamma=lambda t: 1e-4 * 0.9999 ** t)
        x0 = 2
        tester.fedavg(x0, scaled=True)
        tester.fedavg(x0, scaled=False)
