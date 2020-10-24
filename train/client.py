import _init_paths
import os
import sys
import pickle
from typing import Dict, Any, Callable, Tuple

import torch
import torchvision
from torch import optim, nn
from torch.utils.data import sampler
from torch.utils.tensorboard import SummaryWriter

import common
from train.model import Net
import utils.logger

from utils.widgets import init_pars, data_b_to_right_device, get_output
import os.path as osp
from copy import deepcopy
from common import *
from utils.widgets import test_loader
from gensim.scripts.glove2word2vec import glove2word2vec
import threading


class Client:
    batch_size_warn_count = 0
    """
    Warning: dataiter will not be logged into checkpoints
    """
    def __init__(self, train_set, train_indices, is_available, collate_fn=None, **kwargs):
        """
        :param train_set: the whole dataset
        :param train_indices:
            list (indices of the train images)
            str (file path recording the indices, produced by partition)
        :param is_available: a function to judge whether the client is available
        :param kwargs:
            train_batch_size
            id
        """
        self.available_blocks = None
        self.num_hours1block = None
        self.is_available_input = 'round'
        if isinstance(is_available, tuple):
            if is_available[0] == 'check_time':
                self.available_blocks = is_available[1]
                self.num_hours1block = is_available[2]
                is_available = self.check_time
                self.is_available_input = 'time'
            elif is_available[0] == 'check_in_range':   # for modeled_mid
                self.available_start, self.available_end = is_available[1]      # time2int output
                is_available = self.check_in_range
                self.is_available_input = 'time'
            else:
                raise ValueError(f"is_available is a tuple that can not be recognized")

        self.is_available = is_available
        self.train_batch_size = kwargs.pop("train_batch_size", 5)
        self.shuffle = kwargs.pop("shuffle", True)
        self.id = kwargs.pop("id", None)
        self.collate_fn = collate_fn

        if len(kwargs) != 0:
            raise ValueError(f"Unexpected parameter {kwargs}")

        if train_indices is None:
            # do not use dataloader, dataset is already partitioned
            self.dataset = train_set
            self.train_indices = None
        else:
            if isinstance(train_indices, list):
                self.train_indices = train_indices
            elif isinstance(train_indices, str):
                with open(train_indices, 'r') as f:
                    self.train_indices = eval(f.readline())
                if not isinstance(self.train_indices, list):
                    raise ValueError(f"{train_indices} is not a valid client file")
            else:
                raise ValueError(f"train_indices with type {type(train_indices)} is not accepted")

            if self.num_samples < self.train_batch_size:
                if Client.batch_size_warn_count == 0:
                    print(
                        f"some clients are given batch size too high, using {self.num_samples} instead for one of them "
                        f"(full batch)")
                Client.batch_size_warn_count += 1
                self.train_batch_size = self.num_samples

            if self.shuffle:
                _sampler = sampler.SubsetRandomSampler(self.train_indices)
            else:
                _sampler = sampler.SequentialSampler(self.train_indices)
            self.dataloader = torch.utils.data.DataLoader(train_set, batch_size=self.train_batch_size,
                                                          collate_fn=collate_fn, sampler=_sampler)

        self.__last_grad_sum = None  # cache the update calculated in last training
        self.__train_log = []  # log real iteration where self is chosen to train in it

    def check_time(self, current_time):
        # current_time: timedelta, only care seconds (days ignored)
        hour = current_time.seconds // 3600
        block_idx = hour // self.num_hours1block
        return block_idx in self.available_blocks

    def check_in_range(self, current_time):
        # current_time: timedelta, only care seconds (days ignored)
        seconds = current_time.seconds
        if self.available_end == self.available_start:
            return True
        if self.available_end > self.available_start:
            return self.available_start <= seconds <= self.available_end
        else:       # for available_end == available_start: always available
            return seconds >= self.available_end or seconds <= self.available_start

    @property
    def train_log(self):
        return self.__train_log

    @property
    def num_samples(self):
        return len(self.train_indices) if self.train_indices is not None else len(self.dataset)

    @property
    # last iteration when the client is picked -1 if never picked
    def r(self):
        try:
            return self.__train_log[-1]
        except IndexError:
            return -1

    @property
    def latest_grad(self):
        return self.__last_grad_sum

    def get_loss(self, net, criterion=nn.CrossEntropyLoss(reduction='sum')):
        with torch.no_grad():
            loss = 0.0
            num_samples = 0
            for inputs, labels in self.dataloader:
                if inputs.shape[0] == 1:  # can't resolve batchnorm
                    continue

                inputs = inputs.to(device=device)
                labels = labels.to(device=device)

                outputs = net(inputs)
                loss += criterion(outputs, labels).item()
                num_samples += len(labels)
        return loss / num_samples

    def train(self, current_model, criterion, lr, scale, C, current_round, abs_grad_sum_=None, mu_FedProx=0.0,
              scale_in_it=False):
        """
        won't modify model, only return information about update
        :param scale_in_it: True to scale in iteration (use C * scale iterations)
        :param mu_FedProx: proxy term for FedProx
        :param abs_grad_sum_: holder to pass the actual update value (for FedAvg)
        :param lr: learning rate
        :param current_round: current round in the training process
        :param C: number of local iterations
        :param scale: scale of the loss function for unbalanced data (N * num_samples / NUM_CIFAR10_TRAIN)
        :param criterion: criterion to be used
        :param current_model: model to be trained
        :return: the difference between 2 successive ***scaled gradient sum (-update / base_lr)***
        """
        model = deepcopy(current_model).to(device=device)
        initial_model = deepcopy(current_model).to(device=device)
        for par in initial_model.parameters():
            par.requires_grad = False
        optimizer = optim.SGD(model.parameters(), lr=lr)

        # FedLaAvg and first round
        if abs_grad_sum_ is None and self.__last_grad_sum is None:
            self.__last_grad_sum = [torch.zeros_like(par.data) for par in model.parameters()]

        if scale_in_it:
            C_ = max(1, round(C * scale))
            scale_ = 1
        else:
            C_ = C
            scale_ = scale

        if self.train_indices is None:
            if not self.shuffle:
                raise ValueError("We should shuffle the dataset for hard partitioned clients")
            dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.train_batch_size,
                                                     collate_fn=self.collate_fn, shuffle=True)
        else:
            dataloader = self.dataloader
        dataiter = iter(dataloader)
        for c_it in range(C_):
            try:
                inputs, labels = dataiter.__next__()
                assert len(labels) != 1
            except StopIteration as e:
                dataiter = iter(dataloader)
                inputs, labels = dataiter.__next__()
            except AssertionError as e:
                # batchnorm do not accept 1-sample batch
                dataiter = iter(dataloader)
                inputs, labels = dataiter.__next__()

            # for rnn may be not Tensor
            if isinstance(inputs, torch.Tensor):
                if inputs.shape[0] == 1:  # can't resolve batchnorm
                    continue

            optimizer.zero_grad()

            inputs = data_b_to_right_device(inputs)
            labels = labels.to(device=common.device)

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = get_output(model, inputs)
            loss = criterion(outputs, labels) * scale_

            prox = 0.0
            for par, init_par in zip(model.parameters(), initial_model.parameters()):
                prox += mu_FedProx / 2.0 * ((par - init_par) ** 2).sum()
            total_loss = loss + prox
            total_loss.backward()

            for par in model.parameters():
                if torch.isnan(par.grad).sum() != 0:
                    # may cause bug if the exception is caught because the function has modified self ()
                    # but I check the code and think there is no bug
                    # raise RuntimeError("nan in the optimization")
                    print(f"nan in the optimization of client {self.id}, force to 0")
                    par.grad[torch.isnan(par.grad)] = 0.0

            optimizer.step()

        self.__train_log.append(current_round)

        new_grad_sum = [-(new_par.data - old_par.data) / lr for new_par, old_par in
                        zip(model.parameters(), current_model.parameters())]
        if abs_grad_sum_ is not None:
            abs_grad_sum_[0] = new_grad_sum
        else:
            grad_sum_diff = [(new_gs.data - old_gs.data.to(device=device) ) for new_gs, old_gs in
                             zip(new_grad_sum, self.__last_grad_sum)]

            self.__last_grad_sum = [gs.data.cpu() for gs in new_grad_sum]

            return grad_sum_diff

    def state_dict(self):
        state_dict = dict(
            train_log=self.__train_log,
            last_grad_sum=self.__last_grad_sum,
        )
        return state_dict

    def load_state_dict(self, state_dict, load_last_grad_sum=True):
        # to save memory, do not load last_grad_sum for fedavg (current code actually do not save this,
        # but older code save this data, for compatibility this is passed
        self.__train_log = state_dict['train_log']
        if load_last_grad_sum:
            self.__last_grad_sum = state_dict['last_grad_sum']


def _test():
    train_set = torchvision.datasets.CIFAR10(root=raw_data_dir, train=True, download=True)
    client_partition = "N1000-balance"
    train_indices1 = osp.join(data_cache_dir, client_partition, '0')
    train_indices2 = osp.join(data_cache_dir, client_partition, '100')
    client1 = Client(train_set, train_indices1, None, train_batch_size=50)
    client2 = Client(train_set, train_indices2, None, train_batch_size=50)
    run_name = "2client_example"
    sub_run_name = 'async'
    run_name = run_name + '_' + sub_run_name
    if sub_run_name == '1local':
        C1, C2 = 1, 1
        sc1, sc2 = 1.0, 1.0

        def lr(r):
            return 1e-2
    elif sub_run_name == 'sync':
        C1, C2 = 3, 3
        sc1, sc2 = 1.0, 1.0

        def lr(r):
            return 1e-2
    elif sub_run_name == 'sync_bias1':
        C1, C2 = 3, 3
        sc1, sc2 = 5.0, 1.0

        def lr(r):
            return 2e-3
    elif sub_run_name == 'sync_bias2':
        C1, C2 = 3, 3
        sc1, sc2 = 1.0, 5.0

        def lr(r):
            return 2e-3
    elif sub_run_name == 'async':
        C1, C2 = 1, 5
        sc1, sc2 = 1.0, 1.0

        def lr(r):
            return 2e-3
    elif sub_run_name == 'async_scaled':
        C1, C2 = 1, 5
        sc1, sc2 = 5.0, 1.0

        def lr(r):
            return 2e-3 if r <= 1000 else 2e-4
    else:
        raise ValueError()

    logger = utils.logger.Logger(run_name)
    num_total_samples = client1.num_samples + client2.num_samples
    ratio1, ratio2 = client1.num_samples / num_total_samples, client2.num_samples / num_total_samples
    scale1, scale2 = ratio1 * 2, ratio2 * 2

    net: nn.Module = Net()
    net.to(device=device)

    if logger.has_checkpoint():
        logger.load_clients([client1, client2])
        c_round = logger.load_checkpoint_round()
        logger.load_model(net)
    else:
        c_round = 0
        print("no checkpoint")

    for round_ in range(c_round, 10000):
        lr_ = lr(round_)
        grad_sum = [torch.zeros_like(par) for par in net.parameters()]
        grad_sum_ = [[torch.zeros_like(par) for par in net.parameters()]]
        client1.train(net, criterion=nn.CrossEntropyLoss(), lr=lr_, scale=scale1, C=C1, current_round=round_,
                      abs_grad_sum_=grad_sum_)
        grad_sum = [gs + d / 2 * sc1 for gs, d in zip(grad_sum, grad_sum_[0])]
        client2.train(net, criterion=nn.CrossEntropyLoss(), lr=lr_, scale=scale2, C=C2, current_round=round_,
                      abs_grad_sum_=grad_sum_)
        grad_sum = [gs + d / 2 * sc2 for gs, d in zip(grad_sum, grad_sum_[0])]

        with torch.no_grad():
            for par, gs in zip(net.parameters(), grad_sum):
                par.data -= gs * lr_
        acc1_, acc2_ = [-1.], [-1.]
        loss1_, loss2_ = [-1.], [-1.]
        grad_norm1_, grad_norm2_ = [-1.], [-1.]
        test_loader(net, client1.dataloader, acc_=acc1_, loss_=loss1_, grad_norm_=grad_norm1_)
        test_loader(net, client2.dataloader, acc_=acc2_, loss_=loss2_, grad_norm_=grad_norm2_)
        c_accuracy = acc1_[0] * ratio1 + acc2_[0] * ratio2
        c_loss = loss1_[0] * ratio1 + loss2_[0] * ratio2
        # print(round_)
        if round_ % 1 == 0:
            logger.add_scalar('acc1', acc1_[0], round_, write_file=False)
            logger.add_scalar('acc2', acc2_[0], round_, write_file=False)
            logger.add_scalar('acc', c_accuracy, round_, write_file=False)
            logger.add_scalar('loss1', loss1_[0], round_, write_file=False)
            logger.add_scalar('loss2', loss2_[0], round_, write_file=False)
            logger.add_scalar('loss', c_loss, round_, write_file=False)
            logger.add_scalar('grad_norm1', grad_norm1_[0], round_, write_file=False)
            logger.add_scalar('grad_norm2', grad_norm2_[0], round_, write_file=False)
            print(f"round: {round_}:")
            print(f"acc {c_accuracy}, acc1 {acc1_[0]}, acc2 {acc2_[0]}")
            print(f"loss {c_loss}, loss1 {loss1_[0]}, loss2 {loss2_[0]}")
            print(f"grad_norm1 {grad_norm1_[0]}, grad_norm2 {grad_norm2_[0]}")

    logger.dump_model(net)
    logger.dump_clients([client1, client2])
    logger.dump_meta({'current_round':c_round})


if __name__ == "__main__":
    _test()
