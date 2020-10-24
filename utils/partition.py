import _init_paths
import random
import os.path as osp
import os
import pickle as pkl
from math import floor

import common
import numpy as np

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def partition_class(train_set, shuffle=True):
    """
    :param train_set: pytorch dataset to be partitioned mnist or cifar10
    :param shuffle: whether to shuffle data in one class
    :return inds: inds[label] is the list of image index with the given label
    """
    inds = [[] for _ in range(common.NUM_CIFAR10_CLASSES)]

    for i, (fig, label) in enumerate(train_set):
        inds[label].append(i)

    if shuffle:
        for inds_i in inds:
            random.shuffle(inds_i)

    return inds


def train_set2files(raw_data_dir=common.raw_data_dir, output_data_dir=osp.join(common.data_cache_dir, "processed")):
    if osp.exists(output_data_dir):
        raise ValueError(f'Output data directory \"{output_data_dir}\" exists!')

    train_set = torchvision.datasets.CIFAR10(root=raw_data_dir, train=True, download=True,
                                             transform=common.transform)
    os.makedirs(output_data_dir, exist_ok=True)
    for ind, data in enumerate(train_set):
        with open(osp.join(output_data_dir, str(ind)), 'wb') as f:
            pkl.dump(data, f)


def partition(raw_data_dir=common.raw_data_dir, output_data_dir=osp.join(common.data_cache_dir, 'debug'),
              num_clients=100, **kwargs):
    """
    Partition data and save to output_data_dir
        Each class with num_clients // NUM_CIFAR10_CLASSES clients
        Each client is a txt file named by client index, while the content is image index array (format by f"{arr}")
        The txt file can be read into a python object by eval the first line
        client idx are sorted by their corresponding image class, e.g. 0-99 for class 0, 100-199 for class 1
    :param output_data_dir: the directory to save the partitioned data
    :param raw_data_dir: the directory with the raw data
    :param num_clients: number of clients, should be multiple of 10 (CIFAR10 class number)
    :param kwargs:
        num_samples_base: control the degree of data imbalance, larger means imbalance lower
        num_samples_clamp_thres: control the degree of data imbalance, trivial
        seed: random process seed, to enforce fairness
        dataset: cifar10 / mnist
        num_iid: num_iid samples are taken IID from the dataset, default 0 (each class 0 / 10 images)
    """
    num_samples_base = kwargs.pop("num_samples_base", 5)
    num_samples_clamp_thres = kwargs.pop("num_samples_clamp_thres", 2)
    seed = kwargs.pop("seed", 1)
    dataset_indicator = kwargs.pop("dataset", "cifar10")
    num_iid = kwargs.pop("num_iid", 0)

    random.seed(seed)
    np.random.seed(seed)

    if len(kwargs) != 0:
        raise ValueError(f"Unexpected parameter {kwargs}")
    if num_iid % 10 != 0:
        raise ValueError(f"Expect num_iid to be multiple of 10, but get {num_iid}")

    # parameter check
    if num_samples_base <= num_samples_clamp_thres:
        raise ValueError(f"num_samples_base \"{num_samples_base}\" should be larger than num_samples_clamp_thres")
    if num_clients % common.NUM_CIFAR10_CLASSES != 0:
        raise ValueError(f'num_clients \"{num_clients}\" should be a multiple of NUM_CIFAR10_CLASSES')
    if osp.exists(output_data_dir):
        raise ValueError(f'Output data directory \"{output_data_dir}\" exists!')
    else:
        os.makedirs(output_data_dir)

    if dataset_indicator == 'cifar10':
        train_set = torchvision.datasets.CIFAR10(root=raw_data_dir, train=True, download=False)
        num_clients1class = num_clients // common.NUM_CIFAR10_CLASSES
    elif dataset_indicator == "mnist":
        # note that each digit has different number of images
        train_set = torchvision.datasets.MNIST(root=raw_data_dir, train=True, download=False)
        num_clients1class = num_clients // common.NUM_MNIST_CLASSES
    else:
        raise ValueError(f"dataset = {dataset_indicator} is not recognized")

    inds = partition_class(train_set)

    num_iid1class = num_iid // 10
    num_non_iid_start = num_iid1class * num_clients

    for class_ in range(common.NUM_CIFAR10_CLASSES):
        # unbalance partition
        while True:
            num_samples_list = num_samples_base + np.random.randn(num_clients1class)
            num_samples_list[num_samples_list < num_samples_base - num_samples_clamp_thres] = \
                num_samples_base - num_samples_clamp_thres
            num_samples_list[num_samples_list > num_samples_base + num_samples_clamp_thres] = \
                num_samples_base + num_samples_clamp_thres
            sum_weight = num_samples_list.sum()
            num_samples_list /= sum_weight  # normalize

            pre_weight = 0.0
            accumulated = [0.0]
            for end in range(num_clients1class):
                pre_weight = pre_weight + num_samples_list[end]
                accumulated.append(pre_weight)
            accumulated[-1] = 1.0
            accumulated = np.round(np.array(accumulated) * len(inds[class_])).astype(int)

            num_samples_list = [accumulated[i + 1] - accumulated[i] for i in range(num_clients1class)]
            num_samples_list = np.array(num_samples_list)
            if not (num_samples_list < num_iid).sum() == 0:
                print(f'In class {class_}, client with less than {num_iid} data exists, retrying...')
            else:
                break

        start_train = num_non_iid_start
        for worker_ind in range(num_clients1class):
            worker_identifier = worker_ind + num_clients1class * class_
            worker_fp = osp.join(output_data_dir, str(worker_identifier))

            num_train_samples = num_samples_list[worker_ind]

            worker_ind_global = class_ * num_clients1class + worker_ind
            data_inds_iid = [class_inds[worker_ind_global * num_iid1class + idx]
                             for class_inds in inds for idx in range(num_iid1class)]
            data_inds_non_iid = inds[class_][start_train: start_train + num_train_samples - num_iid]

            file_content = f"{data_inds_iid + data_inds_non_iid}"
            with open(worker_fp, 'w') as f:
                f.write(file_content + '\n')
                f.write(f"number of training samples: {num_train_samples}\n")
                f.write(f"samples are from class {class_}\n")

            start_train += num_train_samples - num_iid


def _test():
    # train_set2files()
    # partition(num_clients=1000, output_data_dir=osp.join(common.data_cache_dir, 'cifar10_N1000_ni0'), dataset="cifar10")
    # partition(num_clients=1000, output_data_dir=osp.join(common.data_cache_dir, 'mnist_N1000_ni0'), dataset="mnist")
    # partition(num_clients=1000, output_data_dir=osp.join(common.data_cache_dir, 'cifar10_N1000_ni20'), num_iid=20, dataset="cifar10")
    # partition(num_clients=1000, output_data_dir=osp.join(common.data_cache_dir, 'mnist_N1000_ni20'), num_iid=20, dataset="mnist")
    N = 200
    ni = 0
    partition(num_clients=N, output_data_dir=osp.join(common.data_cache_dir, f'mnist_N{N}_ni{ni}'), num_iid=ni, dataset="mnist")


if __name__ == '__main__':
    _test()

