import _init_paths
import math
import os
import random

import torchvision
import pickle as pkl
import numpy as np

from torch import nn

import common
from common import *

from utils.logger import Logger
from train.model import Net
from train.client import Client
from utils.widgets import init_pars, get_spare_dir

from utils.partition import partition_class


def diurnal_available(E, alpha, client_group):
    """
    :param E: defined in paper
    :param alpha: defined in paper, ratio
    :param client_group: 0 or 1, 0: available for E rounds 1: available for \alpha E rounds
    :return: the availability function to be passed to a Client initializer
        prototype: is_available(r) -> bool
    """

    def is_available(r):
        E0 = int(E)
        E1 = int(alpha * E)
        period = E0 + E1
        r_mod = r % period
        if client_group == 0:
            return r_mod < E0
        elif client_group == 1:
            return r_mod >= E0
        else:
            raise ValueError(f"invalid client_group: {client_group}")

    return is_available


def block_available(E, num_blocks, client_block):
    """
    :param E: defined in paper
    :param num_blocks: total number of blocks
    :param client_block: the block idx for this client
    :return: the availability function to be passed to a Client initializer
        prototype: is_available(r) -> bool
    """

    def is_available(r):
        r_in_period = r % E
        num_rounds1block = E // num_blocks
        current_block = r_in_period // num_rounds1block
        return current_block == client_block

    return is_available


def initialize_clients(trainset, client_partition, E, **kwargs):
    """
    :param trainset: training set
    :param client_partition: the directory saving client data partition schemes
    :param E: defined in paper
    :param kwargs:
        alpha: defined in the paper
        train_batch_size: training batch size for each client
        strategy: chosen from ('time', 'number')
            'time' means available for alpha E rounds
            'number' means alpha proportion of clients are available in half time
                e.g. alpha = 0.2, 20% clients (those corresponding to class 1 and 2) are available in the first block
                for number, alpha = 0.5 means balanced
            'mix' means first group has (1 - alpha) with class 0-4 and alpha with class 5-9, while the second group is the contrast
            'block' means partition multiple client block with different class
        seed: remove randomness among experiments
        num_blocks: number of blocks for block strategy
    :return: a list of the clients
    """
    alpha = kwargs.pop("alpha", 0.5)
    train_batch_size = kwargs.pop("train_batch_size", 5)
    strategy = kwargs.pop('strategy', 'time')
    seed = kwargs.pop('seed', common.other_seed)
    num_blocks = kwargs.pop("num_blocks", 5)

    random.seed(seed)

    N = None
    for _, _, _files in os.walk(client_partition):
        N = len(_files)
        break
    if N == 0:
        raise ValueError(f"invalid client_partition directory: {client_partition}")
    if N % 2 != 0:
        raise ValueError(f"N = {N} should be multiple of 2")
    if N % num_blocks != 0:
        raise ValueError(f"N={N} should be a multiple of num_blocks={num_blocks}")
    if E % num_blocks != 0:
        raise ValueError(f"E={E} should be a multiple of num_blocks={num_blocks}")

    clients = []
    if strategy in ('time', 'number'):
        if strategy == 'time':
            num_classes_in_day = 5
            alpha_time = alpha
        elif strategy == 'number':
            _eps = 0.0001
            if _eps < alpha % 0.1 < 0.1 - _eps:
                raise ValueError(f"When strategy == number, alpha={alpha}, but should be multiple of 0.1")
            num_classes_in_day = round(10 * alpha)
            alpha_time = 1
        else:
            raise RuntimeError("bug in program")

        for client_idx in range(N):
            if client_idx < N // 10 * num_classes_in_day:
                client_group = 0
            else:
                client_group = 1
            client = Client(trainset, osp.join(client_partition, str(client_idx)),
                            diurnal_available(E, alpha_time, client_group),
                            train_batch_size=train_batch_size, id=client_idx)
            clients.append(client)
    elif strategy == 'mix':
        num_clients_minor = round(N // 2 * alpha)
        num_clients_major = N // 2 - num_clients_minor
        group0_client_idxs = \
            random.sample(range(N//2), num_clients_major) + random.sample(range(N//2, N), num_clients_minor)
        group0_client_idxs = set(group0_client_idxs)
        for client_idx in range(N):
            client_group = 0 if client_idx in group0_client_idxs else 1
            client = Client(trainset, osp.join(client_partition, str(client_idx)),
                            diurnal_available(E, 1, client_group),
                            train_batch_size=train_batch_size, id=client_idx)
            clients.append(client)
        random.shuffle(clients)
    elif strategy == 'block':
        num_clients1block = N // num_blocks
        for client_idx in range(N):
            client_block = client_idx // num_clients1block
            client = Client(trainset, osp.join(client_partition, str(client_idx)),
                            block_available(E=E, num_blocks=num_blocks, client_block=client_block),
                            train_batch_size=train_batch_size, id=client_idx)
            clients.append(client)
        random.shuffle(clients)
    else:
        raise ValueError(f"strategy={strategy}, but should be chosen from ('time', 'number')")
    return clients


def _test1():
    trainset = torchvision.datasets.CIFAR10(root=raw_data_dir, train=True, download=False)
    clients = initialize_clients(trainset, osp.join(data_cache_dir, "cifar10_N100_ni0"),
                                 E=10, alpha=0.3, train_batch_size=5, strategy='block')
    training_indices = set()

    from matplotlib import pyplot as plt
    for client_idx, client in enumerate(clients):
        for idx in client.train_indices:
            training_indices.add(idx)
        plt.scatter(client.num_samples, random.random(), c='red')
    plt.savefig(osp.join('debug', 'statistic.png'))
    print(f"total training samples: {len(training_indices)}")

    for round_ in range(30):
        print(f"round: {round_}")
        available_set = []
        for client_idx, client in enumerate(clients):
            if client.is_available(round_):
                available_set.append(client.id)
        print(available_set)


if __name__ == '__main__':
    _test1()
