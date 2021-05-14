import _init_paths

import random

from math import ceil

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp

import torchvision
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pack_sequence, pad_sequence

import common
from utils.din import DeepInterestNetwork, DeepInterestNetworkFixEmb


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()

    def load_initial(self):
        fp = osp.join(common.cache_fd, f'{self.id}.pth')
        self.load_state_dict(torch.load(fp))

    def save_initial(self):
        fp = osp.join(common.cache_fd, f'{self.id}.pth')
        torch.save(self.state_dict(), fp)

    def exists_initial(self):
        fp = osp.join(common.cache_fd, f'{self.id}.pth')
        return osp.exists(fp)


class Net(MyModule):
    def __init__(self):
        super().__init__()
        self.id = "CPCFF"
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Logistic(MyModule):
    def __init__(self):
        super().__init__()
        self.id = "logistic"
        self.fc = nn.Linear(28*28, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc(x)
        return x


class MobileNetV2(MyModule):
    def __init__(self, width_mult=1.0):
        super().__init__()
        self.id = f"mobile2_wm{width_mult}"
        self.net = torchvision.models.mobilenet_v2(num_classes=10, width_mult=width_mult)

    def forward(self, x):
        x = self.net(x)
        return x


class BowLinear(MyModule):
    def __init__(self, num_features):
        super().__init__()
        self.id = f"linear{num_features}"
        self.fc = nn.Linear(num_features, 1)

    def forward(self, x):
        output = self.fc(x)
        output = output.view(-1, 1)
        neg_vec = torch.zeros_like(output)
        return torch.cat((neg_vec, output), 1)


class LSTM2(MyModule):
    def __init__(self, input_size=300, hidden_size=256):
        super().__init__()
        self.id = f"lstm_i{input_size}_h{hidden_size}"
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        self.lstm.flatten_parameters()
        x = self.lstm(x)[0]
        # noinspection PyTypeChecker
        x, lengths = torch.nn.utils.rnn.pad_packed_sequence(x)
        x = x[lengths - 1, range(x.shape[1])]
        x = self.fc(x)
        x = x.view(-1, 1)
        neg_vec = torch.zeros_like(x)
        return torch.cat((neg_vec, x), 1)


class DCNN(MyModule):
    def __init__(self, num_features=300):
        super().__init__()
        self.id = f"dcnn{num_features}"
        self.num_features = num_features
        self.top_k = 4
        self.dmp = DynamicMaxPool(top_k=self.top_k)
        self.ks1 = 7
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=3*num_features, kernel_size=self.ks1,
                               padding=self.ks1-1, groups=num_features//2)
        self.ks2 = (6, 5)
        self.conv2 = nn.Conv2d(in_channels=num_features//2, out_channels=14*num_features//4,
                               kernel_size=self.ks2, padding=(0, self.ks2[1]-1), groups=num_features//4)
        self.fc = nn.Linear(in_features=self.top_k*14*num_features//4, out_features=1)

    def forward(self, x):
        # batch_size x seq_len x dim
        x, lengths = pad_packed_sequence(x, batch_first=True)
        batch_size = x.shape[0]
        dim = x.shape[2]

        # batch_size x dim x seq_len
        x = x.transpose(1, 2)
        # batch_size x out_channels1 x seq_len+6
        x = self.conv1(x)
        # batch_size x seq_len+6 x out_channels1
        x = x.transpose(1, 2)
        # batch_size x max_pool_dim x out_channels1
        x, pool_result_ranges = self.dmp(x, lengths, 1, 2, pool_ranges=lengths+(self.ks1-1))
        # batch_size x out_channels1 x max_pool_dim
        x = x.transpose(2, 1)
        # batch_size x dim//2 x num_maps x max_pool_dim
        x = x.view(batch_size, dim//2, x.shape[1]//(dim//2), x.shape[-1])
        x = torch.tanh(x)
        # batch_size x ? x 1 x out_channels2
        x = self.conv2(x)
        # batch_size x dim//4 x num_maps x seq_len
        x = x.view(batch_size, dim//4, x.shape[1]//(dim//4), x.shape[-1])
        # batch_size x seq_len x num_maps x dim//4
        x = x.transpose(1, 3)
        # batch_size x max_pool_dim x num_maps x dim//4, aligned
        x, _ = self.dmp(x, lengths, 2, 2, pool_ranges=numpy.array(pool_result_ranges)+(self.ks2[1]-1))
        x = torch.tanh(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        x = x.view(-1, 1)
        neg_vec = torch.zeros_like(x)
        return torch.cat((neg_vec, x), 1)


class DinWhole(MyModule):
    def __init__(self):
        super().__init__()
        self.id = "din-whole"
        self.din = DeepInterestNetwork(activation='PReLU')

    def forward(self, *batch):
        return self.din(*batch)


class DinDiceWhole(MyModule):
    def __init__(self):
        super().__init__()
        self.id = "din-dice-whole"
        self.din = DeepInterestNetwork(activation='Dice')

    def forward(self, *batch):
        return self.din(*batch)


class DinDice(MyModule):
    def __init__(self):
        super().__init__()
        self.id = "din-dice"
        self.din = DeepInterestNetworkFixEmb(DeepInterestNetwork(activation='Dice'), copy_pars=False)

    def forward(self, *batch):
        return self.din(*batch)


class Din(MyModule):
    def __init__(self):
        super().__init__()
        self.id = "din"
        self.din = DeepInterestNetworkFixEmb(DeepInterestNetwork(activation='PReLU'), copy_pars=False)

    def forward(self, *batch):
        return self.din(*batch)


def kmax_pooling(x, k, dim=0):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index)


class DynamicMaxPool(nn.Module):
    def __init__(self, top_k):
        super().__init__()
        self.top_k = top_k

    def forward(self, x, lengths, layer, total_layers, pool_ranges):
        results = []
        for sample, length, pool_range in zip(x, lengths, pool_ranges):
            # pool_range x out_channels
            sample = sample[: pool_range]
            pool_target = max(self.top_k, ceil((total_layers - layer) / total_layers * length))
            sample = kmax_pooling(sample, pool_target, dim=0)
            results.append(sample)
        pool_result_ranges = [len(result) for result in results]
        results = pad_sequence(results, batch_first=True)
        return results, pool_result_ranges


def main():
    from utils import widgets

    torch.random.manual_seed(0)
    random.seed(0)
    numpy.random.seed(0)

    for model in [Logistic(), LSTM2(input_size=25, hidden_size=16)]:
        if model.exists_initial():
            model.load_initial()
            raise FileExistsError(f"{model.id} exists")

        model.save_initial()

        print(f"{widgets.count_parameters(model)} parameters in {model.id}")


if __name__ == '__main__':
    main()
