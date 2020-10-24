from copy import deepcopy

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from utils.widgets import get_spare_dir

TAG = "loss"


class Tester:

    def __init__(self, writer, gamma=1e-4, T=100032, C=10):
        self.functions = [Tester.__quad_func(0), Tester.__quad_func(1), Tester.__quad_func(10), Tester.__quad_func(11)]
        self.day_idxs = [0, 1]
        self.night_idxs = [2, 3]
        self.Is = [50, 150]
        self.gamma = gamma
        self.T = T
        self.writer = writer
        self.C = C

    @staticmethod
    def __quad_func(e):
        return {'ori': lambda x: (x - e) ** 2, 'grad': lambda x: 2 * (x - e)}

    def get_update(self, function, x, mu):
        x_ = deepcopy(x)
        for c in range(self.C):
            x_ -= self.gamma * (function['grad'](x_) + 2 * mu * (x_ - x))
        return x_ - x

    def fedavg(self, x0, mu=0.0):
        x = x0
        for t in range(self.T):
            if t % sum(self.Is) < self.Is[0]:
                function_idxs = self.day_idxs
            else:
                function_idxs = self.night_idxs

            total_update = 0
            for function_idx in function_idxs:
                update = self.get_update(self.functions[function_idx], x, mu=mu)
                total_update += update
            x += (total_update / len(function_idxs))
            self.writer.add_scalar(TAG, x, t)
        return x

    def lastavg(self, x0):
        x = x0
        latest_updates = [0 for _ in self.functions]
        for t in range(self.T):
            if t % sum(self.Is) < self.Is[0]:
                function_idxs = self.day_idxs
            else:
                function_idxs = self.night_idxs

            for function_idx in function_idxs:
                update = self.get_update(self.functions[function_idx], x)
                latest_updates[function_idx] = update
            x += (sum(latest_updates) / len(self.functions))
            self.writer.add_scalar(TAG, x, t)
        return x

    def scaffold(self, x0):
        x = x0
        cs = [0 for _ in self.functions]
        for t in range(self.T):
            if t % sum(self.Is) < self.Is[0]:
                function_idxs = self.day_idxs
            else:
                function_idxs = self.night_idxs

            updates = []
            cc = sum(cs) / len(cs)
            for function_idx in function_idxs:
                x_ = deepcopy(x)
                grads = []
                for c in range(self.C):
                    grad = self.functions[function_idx]['grad'](x_)
                    grads.append(grad)
                    x_ -= self.gamma * (grad - cs[function_idx] + cc)
                cs[function_idx] = sum(grads) / len(grads)
                updates.append(x_ - x)
            x += np.mean(updates)
            self.writer.add_scalar(TAG, x, t)
        return x


if __name__ == '__main__':
    import os.path as osp
    f_dir = 'runs'

    gamma = 1e-4
    T = 10032
    C = 10

    # log_dir = osp.join(f_dir, "fedavg")
    # with SummaryWriter(log_dir) as writer:
    #     tester = Tester(writer, gamma=gamma, T=T, C=C)
    #     x0 = 0
    #     tester.fedavg(x0)
    #
    # log_dir = osp.join(f_dir, "lastavg")
    # with SummaryWriter(log_dir) as writer:
    #     tester = Tester(writer, gamma=gamma, T=T, C=C)
    #     x0 = 0
    #     tester.lastavg(x0)
    #
    # log_dir = osp.join(f_dir, "scaffold")
    # with SummaryWriter(log_dir) as writer:
    #     tester = Tester(writer, gamma=gamma, T=T, C=C)
    #     x0 = 0
    #     tester.scaffold(x0)

    mu = 10000.0
    log_dir = osp.join(f_dir, f"fedprox_{mu}")
    with SummaryWriter(log_dir) as writer:
        tester = Tester(writer, gamma=gamma, T=T, C=C)
        x0 = 0
        tester.fedavg(x0, mu=mu)


