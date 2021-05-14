import argparse
import copy
import datetime
import random
import re
import sys
import multiprocessing
import threading

import _init_paths
import pickle
import time

import torch.utils.data
import torchvision
from torch import nn
from torchvision import models
import numpy as np

import common
from train.client import Client
import train.model
from train.model import Net, BowLinear, LSTM2, DCNN, DynamicMaxPool, Logistic, DinWhole, DinDiceWhole, DinDice, Din

import torch.optim as optim
import os.path as osp

from train.prepare_clients import initialize_clients
from utils import widgets, alibaba, sentiment140
from utils.alibaba import AlibabaDataset, alibaba_train_fp, alibaba_test_fp

from utils.logger import Logger
from utils.partition import partition
from utils.sentiment140 import Sentiment140Dataset, BasicProcess, CleanTweet, BagOfWords, ComposedProcess

from torchvision import transforms


class Server:
    def __init__(self, mode):
        self.mode = mode
        self.grad_sum = None
        self.running_grad_sum = None
        self.latest_grad_sum_avg = None
        self.grad_sum_cache = None
        self.participated_clients = set()  # clients that have participated, hold the ids

    def new_round(self, model):
        if self.mode == 'fedavg':
            self.grad_sum = [torch.zeros_like(par) for par in model.parameters()]
            if self.running_grad_sum is None:
                self.running_grad_sum = [torch.zeros_like(par) for par in model.parameters()]
        elif self.mode == 'lastavg':
            if self.latest_grad_sum_avg is None:
                self.latest_grad_sum_avg = [torch.zeros_like(par) for par in model.parameters()]
        elif self.mode == 'waitavg':
            self.grad_sum_cache = [torch.zeros_like(par) for par in model.parameters()]
            self.participated_clients.clear()
        else:
            raise ValueError(f"Unexpected mode {self.mode}")

    def aggregate(self, client_upload, **kwargs):
        K = kwargs.pop("K", None)
        N = kwargs.pop("N", None)

        if self.mode == 'fedavg':
            for gs, client_gs in zip(self.grad_sum, client_upload):
                gs += client_gs / K
        elif self.mode == 'lastavg':
            for gs, client_gs in zip(self.latest_grad_sum_avg, client_upload):
                gs += client_gs / N
        elif self.mode == 'waitavg':
            for gs, client_gs in zip(self.grad_sum_cache, client_upload):
                gs += client_gs / N
        else:
            raise ValueError(f"Unexpected mode {self.mode}")

    def update(self, model, lr, **kwargs):
        momentum = kwargs.pop("momentum", 0.0)
        if self.mode == 'fedavg':
            for gs, running_gs in zip(self.grad_sum, self.running_grad_sum):
                running_gs[...] = running_gs * momentum + gs * (1 - momentum)
            for running_gs, par in zip(self.running_grad_sum, model.parameters()):
                par.data.sub_(lr * running_gs)
        elif self.mode == 'lastavg':
            for gs, par in zip(self.latest_grad_sum_avg, model.parameters()):
                par.data.sub_(lr * gs)
        elif self.mode == 'waitavg':
            for gs, par in zip(self.grad_sum_cache, model.parameters()):
                par.data.sub_(lr * gs)
        else:
            raise ValueError(f"Unexpected mode {self.mode}")

    @staticmethod
    def pick_clients(clients, K, pick_outdated, current_round):
        # current_round can be current_time for sentiment140
        available_clients = []
        for client in clients:
            if client.is_available(current_round):
                available_clients.append(client)
        if K == 'all':
            return available_clients
        else:
            if len(available_clients) < K:
                print(f"*** Warning: round {current_round} available clients {len(available_clients)} < K {K}")
                K = len(available_clients)
            if pick_outdated:
                available_clients = sorted(available_clients, key=lambda c: c.r)
                picked_clients = available_clients[:K]
            else:
                picked_clients = random.sample(available_clients, K)
            return picked_clients

    def pick_clients_no_participated(self, clients, current_round):
        # pick clients that have not participated in current 'epoch' training
        picked_clients = []
        for client in clients:
            client: Client
            if client.is_available(current_round):
                if client.id not in self.participated_clients:
                    picked_clients.append(client)
                    self.participated_clients.add(client.id)
        return picked_clients


def train_(run_name, test_loader, train_loader, alg, log_args, log_argv,
           criterion=nn.CrossEntropyLoss(), test_criterion=nn.CrossEntropyLoss(reduction='sum'),
           **kwargs):
    # get parameters
    lr = kwargs.pop("lr", lambda r: 1e-2)
    num_rounds = kwargs.pop("num_rounds", 50000) + 1
    model = kwargs.pop("model_ori_path")       # model rather than path
    print_every = kwargs.pop("print_every", 1)
    tb_every = kwargs.pop("tb_every", 1)
    statistics_every = kwargs.pop("statistics_every", 100)
    checkpoint_every = kwargs.pop("checkpoint_every", 100)
    max_checked = kwargs.pop("max_checked", 10000)
    reserve_checkpoint_steps = kwargs.pop("reserve_checkpoint_steps", [])
    E = kwargs.pop("E", None)
    num_non_update_rounds = kwargs.pop("num_non_update_rounds", None)
    log_auc = kwargs.pop("log_auc", False)

    # initialize
    model = model.to(device=common.device)
    logger = Logger(run_name)
    if E is not None:
        time1round = datetime.timedelta(days=1) / E
    else:
        time1round = datetime.timedelta(seconds=0)          # not logged

    # common checkpoint practice
    if logger.has_checkpoint():
        state_dict = logger.load_meta()
        current_round = state_dict['current_round']
        time_base = state_dict['time']
        if 'simulated_time' not in state_dict:
            print("*** Warning: checkpoint of older runs, no simulated time dumped ***")
            simulated_time = datetime.timedelta(seconds=0)
        else:
            simulated_time = state_dict['simulated_time']
        logger.load_model(model)
    else:
        simulated_time = datetime.timedelta(seconds=0)
        current_round = 0
        time_base = 0.0
    start_time = time.time()

    # custom parameters
    if alg == 'sgd':
        num_its1round = kwargs.pop('num_its1round', None)
        train_iter = iter(train_loader)
    elif alg == 'gd':
        num_training_samples = kwargs.pop('num_training_samples')
    elif alg in ('fedavg', 'lastavg', 'waitavg'):
        server: Server = kwargs.pop('server')
        clients = kwargs.pop('clients')
        num_training_samples = sum([client.num_samples for client in clients])
        print(f"Actual number of training samples: {num_training_samples}")
        K = kwargs.pop("K", 100)
        C = kwargs.pop("C", 10)
        scale_in_it = kwargs.pop("scale_in_it", False)
        if logger.has_checkpoint():
            server = logger.load_server()
            logger.load_clients(clients, target_alg=alg)
        if server.mode != alg:
            raise ValueError(f"Wrong server with mode {server.mode}, expected {alg}")
        if alg == 'fedavg':
            momentum = kwargs.pop("momentum", 0.0)
            pick_outdated = kwargs.pop('pick_outdated', False)
            mu_FedProx = kwargs.pop("mu_FedProx", 0.0)
        elif alg == 'lastavg':
            pick_outdated = kwargs.pop('pick_outdated', True)
    if len(kwargs) != 0:
        raise ValueError(f"Unexpected parameter {kwargs}")

    logger.add_meta(log_args, log_argv, current_round)

    num_updates = 0  # for waitavg
    if alg == 'waitavg':
        server.new_round(model)  # first round
    while True:
        lr_ = lr(current_round)
        c_time = time.time()

        if current_round in reserve_checkpoint_steps:
            # to corporate with multistep decay, reserve checkpoint before training
            print("reserve checkpoint")
            logger.reserve_checkpoint()
        if current_round % print_every == 0 and alg != 'gd':
            print(f"{run_name[:50]}, {current_round}, {simulated_time.days}d, lr %.5f" % lr_, end='\r')
        if current_round % tb_every == 0:
            test_acc_, train_acc_ = [-1.], [-1.]
            test_loss_, train_loss_ = [-1.], [-1.]
            test_grad_norm_, train_grad_norm_ = [-1.], [-1.]
            if log_auc:
                test_auc_, train_auc_ = [-1.], [-1.]
            else:
                test_auc_, train_auc_ = None, None
            widgets.test_loader(model, test_loader, max_checked=max_checked, criterion=test_criterion,
                                acc_=test_acc_, loss_=test_loss_, grad_norm_=test_grad_norm_, auc_=test_auc_)
            time_stamp = c_time - start_time + time_base
            logger.add_scalar("test_accuracy", test_acc_[0], current_round, time_stamp)
            logger.add_scalar("test_loss", test_loss_[0], current_round, time_stamp)
            logger.add_scalar("test_grad_norm", test_grad_norm_[0], current_round, time_stamp)
            if log_auc:
                logger.add_scalar("test_auc", test_auc_[0], current_round, time_stamp)
            if alg != 'gd':
                # gd log every iteration since there is no overhead
                widgets.test_loader(model, train_loader, max_checked=max_checked, criterion=test_criterion,
                                    acc_=train_acc_, loss_=train_loss_, grad_norm_=train_grad_norm_, auc_=train_auc_)
                logger.add_scalar("train_accuracy", train_acc_[0], current_round, time_stamp)
                logger.add_scalar("train_loss", train_loss_[0], current_round, time_stamp)
                logger.add_scalar("train_grad_norm", train_grad_norm_[0], current_round, time_stamp)
                if log_auc:
                    logger.add_scalar('train_auc', train_auc_[0], current_round, time_stamp)
            if alg == 'lastavg':
                # log the difference between the real latest gradient average and the running one
                latest_grad_avg = widgets.latest_gradient_avg(clients)
                rel_error, gs_error_norm = widgets.rel_and_norm_error(latest_grad_avg, server.latest_grad_sum_avg)
                logger.add_scalar("gs_rel_error", rel_error, current_round, time_stamp)
                logger.add_scalar("gs_error_norm", gs_error_norm, current_round, time_stamp)
            print("%d, te_ac: %.3f, tr_ac: %.3f, te_lo: %.5f, tr_lo: %.5f, te_gn: %.5f, tr_gn: %.5f" %
                  (current_round, test_acc_[0], train_acc_[0], test_loss_[0], train_loss_[0],
                   test_grad_norm_[0], train_grad_norm_[0]), end='')
            if log_auc:
                print(", te_auc: %.3f, tr_auc: %.3f" %
                      (test_auc_[0], train_auc_[0]), end='')
            print()

        # perform a round
        if alg == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr_)
            running_loss = 0.0
            for c_it in range(num_its1round):
                optimizer.zero_grad()
                try:
                    data_b, label_b = train_iter.__next__()
                except StopIteration as e:
                    train_iter = iter(train_loader)
                    data_b, label_b = train_iter.__next__()
                data_b = widgets.data_b_to_right_device(data_b)
                label_b = label_b.to(device=common.device)
                output_b = widgets.get_output(model, data_b)
                loss = criterion(output_b, label_b)
                loss.backward()
                running_loss += loss.item()
                optimizer.step()
            running_loss /= num_its1round
        elif alg == 'gd':
            optimizer = optim.SGD(model.parameters(), lr_ / num_training_samples)
            optimizer.zero_grad()
            running_loss = 0.0
            num_right = 0
            for b_idx, (data_b, label_b) in enumerate(train_loader):
                print(f"{b_idx}/{len(train_loader)}", end='\r')
                data_b = widgets.data_b_to_right_device(data_b)
                label_b = label_b.to(device=common.device)
                output_b = widgets.get_output(model, data_b)
                with torch.no_grad():
                    pred_b = torch.argmax(output_b, dim=1)
                    num_right += (pred_b == label_b).sum().item()
                loss = criterion(output_b, label_b) * len(label_b)
                loss.backward()
                running_loss += loss.item()
            grad_norm_squ = 0.0
            for par in model.parameters():
                grad_norm_squ += torch.norm(par.grad / num_training_samples) ** 2
            grad_norm = grad_norm_squ ** 0.5
            optimizer.step()
            running_loss /= num_training_samples
            train_acc = num_right / num_training_samples
            time_stamp = c_time - start_time + time_base
            logger.add_scalar("train_accuracy", train_acc, current_round, time_stamp)
            logger.add_scalar("train_loss", running_loss, current_round, time_stamp)
            logger.add_scalar("train_grad_norm", grad_norm, current_round, time_stamp)
            print("GD %d, lr: %.7f, train_acc: %.5f, train_loss: %.5f, train_gn: %.5f" %
                  (current_round, lr_, train_acc, running_loss, grad_norm))
        elif alg in ('fedavg', 'lastavg'):
            server.new_round(model)
            _isa_input = clients[0].is_available_input
            if _isa_input == 'round':
                _current_round = current_round
            elif _isa_input == 'time':
                _current_round = simulated_time
            else:
                raise RuntimeError(f"clients weirdly initialized with is_available_input = {_isa_input}")
            picked_clients = server.pick_clients(clients, K, pick_outdated=pick_outdated,
                                                 current_round=_current_round)

            client: Client
            if alg == 'fedavg':
                for client in picked_clients:
                    scale_inner = len(clients) * client.num_samples / num_training_samples
                    grad_sum_ = [None]
                    client.train(model, criterion, lr_, scale=scale_inner, scale_in_it=scale_in_it,
                                 C=C, current_round=current_round, abs_grad_sum_=grad_sum_, mu_FedProx=mu_FedProx)
                    server.aggregate(grad_sum_[0], K=len(picked_clients))
                server.update(model, lr=lr_, momentum=momentum)
            elif alg == 'lastavg':
                for client in picked_clients:
                    scale_inner = len(clients) * client.num_samples / num_training_samples
                    grad_sum_diff = client.train(model, criterion, lr_, scale=scale_inner, scale_in_it=scale_in_it,
                                                 C=C, current_round=current_round)
                    server.aggregate(grad_sum_diff, N=len(clients))
                if current_round >= num_non_update_rounds:
                    server.update(model, lr=lr_)
        elif alg == 'waitavg':
            _isa_input = clients[0].is_available_input
            if _isa_input == 'round':
                _current_round = current_round
            elif _isa_input == 'time':
                _current_round = simulated_time
            else:
                raise RuntimeError(f"clients weirdly initialized with is_available_input = {_isa_input}")
            picked_clients = server.pick_clients_no_participated(clients=clients, current_round=_current_round)

            client: Client
            for client in picked_clients:
                scale_inner = len(clients) * client.num_samples / num_training_samples
                grad_sum_ = [None]
                client.train(model, criterion, lr_, scale=scale_inner, scale_in_it=scale_in_it,
                             C=C, current_round=current_round, abs_grad_sum_=grad_sum_)
                server.aggregate(grad_sum_[0], N=len(clients))

            if len(server.participated_clients) == len(clients):  # all clients participated update model
                server.update(model, lr=lr_)
                num_updates += 1
                print('\nserver updated model in round: %d for the %d times' % (current_round, num_updates))
                server.new_round(model)
        else:
            raise ValueError(f"Unrecognized alg: {alg}")

        current_round += 1
        simulated_time += time1round
        if current_round % checkpoint_every == 0:
            print("checkpoint")
            logger.flush()
            logger.backup_checkpoint()
            logger.dump_meta({'current_round': current_round,
                              'time': time.time() - start_time + time_base,
                              'simulated_time': simulated_time})
            logger.dump_model(model)
            if alg in ('fedavg', 'lastavg', 'waitavg'):
                logger.dump_server(server)
                logger.dump_clients(clients)
            logger.remove_backup()
        if alg in ('fedavg', 'lastavg', 'waitavg') and current_round % statistics_every == 0:
            logger.add_statistics(clients, current_round)
        if current_round == num_rounds:
            break


def main():
    """
    *** maintenance guidance ***
    When modifying parser:
        please parse it and get the argument into local variables
        please check whether we need to add it to the auto_generated run_name
        please pass it to train for *all* algorithms
    """
    # configure command line args
    import train.command_line

    # parse args
    args = train.command_line.parser.parse_args()
    # modify default arguments
    if args.lr_strategy == 'multi':
        args.lr_decay = 1.0
    default_model_ori = dict(cifar10='CPCFF', sentiment140='lstm_i300_h256', mnist='logistic', ali='din')
    if args.model_ori is None:
        args.model_ori = default_model_ori[args.dataset]
    default_batch_size = dict(cifar10=5, sentiment140=2, mnist=5, ali=2)
    if args.batch_size == -1:
        args.batch_size = default_batch_size[args.dataset]
    default_alpha = dict(time=0.5, number=0.1, mix=0.3, block=None)
    if args.alpha == -1:
        args.alpha = default_alpha[args.strategy]
    default_filter_clients = dict(sentiment140=40, ali=32)
    if args.filter_clients == -1:
        args.filter_clients = default_filter_clients.get(args.dataset, -1)

    # focused parameters
    dataset_indicator = args.dataset
    run_name = args.run_name
    alg = args.algorithm
    N = args.num_total_clients
    beta = args.beta
    C = args.num_local_iterations
    E = args.E
    momentum = args.momentum
    mu_FedProx = args.mu_FedProx
    availability_file = args.availability_file
    # learning rate parameters
    lr_strategy = args.lr_strategy
    init_lr = args.init_lr
    lr_decay = args.lr_decay
    lr_dstep = args.lr_dstep
    lr_indicator = args.lr_indicator
    lr_config = args.lr_config
    # parameters controlling logging
    print_every = args.print_every
    tb_every = args.tb_every
    checkpoint_every = args.checkpoint_every
    sta_every = args.statistics_every
    max_test = args.max_test
    # uncared parameters
    batch_size = args.batch_size
    batch_size_when_test = args.batch_size_when_test
    num_rounds = args.num_rounds
    num_non_update_rounds = args.num_non_update_rounds
    balanced = args.balanced
    shuffle = args.shuffle
    filter_clients = args.filter_clients
    filter_clients_up = args.filter_clients_up
    scale_in_it = args.scale_in_it
    # image classification parameters
    alpha = args.alpha
    normalize_mean = args.normalize_mean
    normalize_std = args.normalize_std
    num_iid = args.num_iid
    strategy = args.strategy
    # ML model
    model_ori = args.model_ori
    nlp_algorithm = args.nlp_algorithm
    glove_model = args.glove_model
    # sentiment140
    availability_model = args.availability_model
    force_variation = args.force_variation
    fv_num_blocks = args.fv_num_blocks
    fv_min_prop = args.fv_min_prop
    fv_max_prop = args.fv_max_prop

    if dataset_indicator == 'sentiment140' and E % 24 != 0:
        raise ValueError("Only accept E=24 for sentiment140")

    # generate args
    if beta != 1.0:
        K = round(N * beta)
    else:
        K = 'all'
    shuffle = True if shuffle else False
    scale_in_it = True if scale_in_it else False
    # run name
    if run_name is None:
        run_name = f"{alg}"
        if dataset_indicator != 'cifar10':
            run_name += f"_{dataset_indicator}"
        if N != 1000:
            run_name += f"_N{N}"
        if beta != 0.1:
            run_name += f"_beta{beta}"
        if C != 10:
            run_name += f"_C{C}"
        if E != 10:
            run_name += f"_E{E}"
        if alpha != default_alpha[strategy] and strategy != 'block':
            run_name += f"_alpha{alpha}"
        if momentum != 0.0:
            run_name += f"_momentum{momentum}"
        if lr_strategy != 'exp':
            run_name += f"_lrS{lr_strategy}"
        if init_lr != 0.01 and lr_strategy != 'multi':
            run_name += f"_lrI{init_lr}"
        if lr_decay != 0.9999 and lr_strategy != 'multi':
            run_name += f"_lrD{lr_decay}"
        if lr_dstep != 1 and lr_strategy != 'multi':
            run_name += f"_lrDs{lr_dstep}"
        if lr_indicator != '?':
            run_name += f"_lrSC{lr_indicator}"   # scheme
        if batch_size != default_batch_size[dataset_indicator]:
            run_name += f"_bs{batch_size}"
        if model_ori != default_model_ori[dataset_indicator]:
            run_name += f"_[mo_{model_ori}]"
        if nlp_algorithm != 'embedding':
            run_name += f"_[nlp_alg_{nlp_algorithm}]"
        if glove_model != 'glove.840B.300d':
            run_name += f'_[gm_{glove_model}]'
        if filter_clients != default_filter_clients.get(dataset_indicator, -1):
            run_name += f"_fc{filter_clients}"
        if filter_clients_up != 2**32:
            run_name += f"_fcu{filter_clients_up}"
        if num_non_update_rounds != 0:
            run_name += f"_nnur{num_non_update_rounds}"
        if availability_model != "blocked":
            run_name += f"_[am_{availability_model}]"
        if normalize_mean != 0.5:
            run_name += f"_nm{normalize_mean}"
        if normalize_std != 0.5:
            run_name += f"_ns{normalize_std}"
        if num_iid != 0:
            run_name += f"_ni{num_iid}"
        if strategy != 'time':
            run_name += f"_[stg_{strategy}]"
        if mu_FedProx != 0.0:
            run_name += f"_mu{mu_FedProx}"
        if force_variation == 1:
            run_name += f"_fv"
        if fv_num_blocks != 24:
            run_name += f"-nb{fv_num_blocks}"
        if fv_min_prop != 0.0:
            run_name += f"-mi{fv_min_prop}"
        if fv_max_prop != 1.0:
            run_name += f"-ma{fv_max_prop}"
        if balanced == 1:
            run_name += f"_balanced"
        if not shuffle:
            run_name += f"_nshuf"
        if scale_in_it:
            run_name += f"_sii"
        if availability_file != "client_available":
            run_name += f"_[_af_{availability_file}]"

    # interact with user
    checkpoint_exists = osp.exists(osp.join(common.record_fd, run_name))
    if checkpoint_exists:
        proceed = input(f"run_name: \"{run_name}\" exists, do you really want to continue from checkpoints? (yes/no)")
        if proceed != 'yes':
            exit()

    # reserve intermediate checkpoints
    reserve_checkpoint_steps = []

    # learning rate
    if lr_strategy == 'const':
        def lr(r):
            return init_lr
    elif lr_strategy == 'exp':
        def lr(r):
            return init_lr * lr_decay ** (r // lr_dstep)
    elif lr_strategy == 'multi':
        if lr_indicator == '?':
            raise ValueError("multistep decay should specify a indicator")
        if lr_decay != 1.0:
            raise ValueError("lr_decay should be 1.0 for multistep decay")

        if lr_config is None:
            def lr(r):
                return init_lr
        else:
            with open(osp.join(common.lr_configure_fd, lr_config + '.txt'), 'r') as f:
                decay_steps = eval(f.readline())
                lrs = eval(f.readline())
                if decay_steps[0] != 0:
                    raise ValueError("invalid lr config since not specify initial lr")
                if len(decay_steps) != len(lrs):
                    raise ValueError("invalid lr config since steps and lrs do not match in length")
            reserve_checkpoint_steps = decay_steps[1:]

            def lr(r):
                for step, lr_ in list(zip(decay_steps, lrs))[::-1]:
                    if r >= step:
                        return lr_
    else:
        raise ValueError(f"Unrecognized lr_strategy: {lr_strategy}")

    # dataset, loaders, and ml model
    model_indicator = model_ori.split('_')[0]
    if model_indicator == "CPCFF":
        ori_model = Net()
    elif model_indicator == 'mobile2':
        width_mult_ = float(model_ori.split('_')[1][2:])
        ori_model = train.model.MobileNetV2(width_mult=width_mult_)
    elif model_indicator == 'logistic':
        ori_model = Logistic()
    elif model_indicator == "lstm":
        _, in_features, h_features = model_ori.split('_')
        ori_model = LSTM2(input_size=int(in_features[1:]), hidden_size=int(h_features[1:]))
    elif model_indicator == 'din-whole':
        ori_model = DinWhole()
    elif model_indicator == 'din-dice-whole':
        ori_model = DinDiceWhole()
    elif model_indicator == 'din-dice':
        ori_model = DinDice()
    elif model_indicator == 'din':
        ori_model = Din()
    else:
        raise ValueError(f"Unrecognized model {model_indicator}")
    ori_model.load_initial()
    if dataset_indicator == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((normalize_mean,) * 3, (normalize_std,) * 3)])
        trainset = torchvision.datasets.CIFAR10(root=common.raw_data_dir, train=True, download=False,
                                                transform=transform)
        testset = torchvision.datasets.CIFAR10(root=common.raw_data_dir, train=False, download=False,
                                               transform=transform)
    elif dataset_indicator == 'mnist':
        transform= transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize((normalize_mean,), (normalize_std,))])
        trainset = torchvision.datasets.MNIST(root=common.raw_data_dir, train=True, download=True,
                                                transform=transform)
        testset = torchvision.datasets.MNIST(root=common.raw_data_dir, train=False, download=True,
                                               transform=transform)
    elif dataset_indicator == 'sentiment140':
        if nlp_algorithm == 'bow':
            bow_num_features = int(re.search(r"[0-9]+", model_ori).group())
            sentiment140_dataset = Sentiment140Dataset(ComposedProcess(
                BasicProcess(),
                CleanTweet(verbose=True),
                BagOfWords(bow_num_features, verbose=True)
            ), embedding_fn=None, transform=BagOfWords.multihot(bow_num_features))
        elif nlp_algorithm == 'embedding':
            sentiment140_dataset = Sentiment140Dataset(ComposedProcess(
                BasicProcess(),
                CleanTweet(verbose=True),
            ), embedding_fn=glove_model, transform="glove_trans")
        else:
            raise ValueError(f"Unexpected nlp_algorithm {nlp_algorithm}")
        print("filtering clients...", end='\r')
        sentiment140_dataset.filter_clients_(filter_clients, up=filter_clients_up)
        sentiment140_dataset.random_select_clients_(N)
        print("clients filtered    ")
        print("drop samples...", end="\r")
        if force_variation:
            sentiment140_dataset.time_variation_(num_blocks=fv_num_blocks, min_prop=fv_min_prop, max_prop=fv_max_prop)
        print("samples droped ")
        print("partitioning dataset...", end='\r')
        trainset, testset = sentiment140_dataset.partition()
        print("dataset partitioned    ")
        actual_N = trainset.count_clients()
        print(f"There are actually {actual_N} clients, {len(trainset)} training samples")
        if actual_N != N:
            response = input(f"After filtering and sampling clients, there are actually {actual_N}!={N} clients, "
                             f"continue? (yes/no)")
            if response != 'yes':
                exit()
    elif dataset_indicator == 'ali':
        print("loading dataset...", end='\r')
        trainset = AlibabaDataset(alibaba_train_fp)
        testset = AlibabaDataset(alibaba_test_fp)
        trainset.shuffle_()
        testset.shuffle_()
        print("dataset loaded    ")
    else:
        raise ValueError(f"Unrecognized dataset {dataset_indicator}")
    criterion_mean = nn.CrossEntropyLoss()
    criterion_sum = nn.CrossEntropyLoss(reduction='sum')
    if dataset_indicator == 'sentiment140' and nlp_algorithm == 'embedding':
        if alg in ('gd', 'sgd'):
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle,
                                                      collate_fn=Sentiment140Dataset.collate_fn)
        else:
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_when_test, shuffle=False,
                                                      collate_fn=Sentiment140Dataset.collate_fn)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_when_test, shuffle=False,
                                                 collate_fn=Sentiment140Dataset.collate_fn)
    elif dataset_indicator == 'ali':
        if alg in ('gd', 'sgd'):
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle,
                                                      collate_fn=AlibabaDataset.collate_fn)
        else:
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_when_test, shuffle=False,
                                                      collate_fn=AlibabaDataset.collate_fn)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_when_test, shuffle=False,
                                                 collate_fn=AlibabaDataset.collate_fn)
    else:
        if alg in ('gd', 'sgd'):
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle)
        else:
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_when_test, shuffle=False)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_when_test, shuffle=False)

    # server and clients
    if alg in ('fedavg', 'lastavg', 'waitavg'):
        server = Server(alg)
        print("getting clients...")
        if dataset_indicator in ('cifar10', 'mnist'):
            partition_fn = f"{dataset_indicator}_N{N}_ni{num_iid}"
            if balanced:
                partition_fn += "_b1"
            partition_fp = osp.join(common.data_cache_dir, partition_fn)
            if not osp.exists(partition_fp):
                if balanced:
                    partition(num_clients=N, num_iid=num_iid, dataset=dataset_indicator,
                              output_data_dir=partition_fp, num_samples_clamp_thres=0.0)
                else:
                    partition(num_clients=N, num_iid=num_iid, dataset=dataset_indicator,
                              output_data_dir=partition_fp)
                print(f"client partition file saved into {partition_fp}")
            clients = initialize_clients(trainset, partition_fp,
                                         E, alpha=alpha, train_batch_size=batch_size, strategy=strategy)
        elif dataset_indicator == 'sentiment140':
            clients = trainset.get_clients(strategy=availability_model, train_batch_size=batch_size)
        elif dataset_indicator == 'ali':
            clients = alibaba.get_clients(osp.join(common.alibaba_fd, f"{filter_clients}_{filter_clients_up}"),
                                          train_batch_size=batch_size, num_clients=N, availability_file=availability_file)
            print("filtering clients")
            trainset.filter_clients_(clients)
            testset.filter_clients_(clients)
            # print("plotting statistics")
            # widgets.plot_time2available_data_ratio(clients)
            # print("plotted")
            print(f"filter complete, {len(trainset)} training samples, {len(testset)} test samples")
            print(f"*** Warning! filter_clients upper and lower bound currently not supported for ali")
        else:
            raise ValueError(f"unrecognized dataset {dataset_indicator}")
        print(f"Got {len(clients)} clients                   ")

    log_auc = dataset_indicator == 'ali'

    # starting training
    if alg == 'sgd':
        print(f"number of iterations one round in SGD: {K * C}")
        train_(run_name=run_name, test_loader=testloader, train_loader=trainloader, alg=alg,
               num_its1round=K * C,
               criterion=criterion_mean, test_criterion=criterion_sum,
               lr=lr, num_rounds=num_rounds, model_ori_path=ori_model,
               print_every=print_every, tb_every=tb_every, checkpoint_every=checkpoint_every, max_checked=max_test,
               log_args=args, log_argv=sys.argv,
               log_auc=log_auc,
               reserve_checkpoint_steps=reserve_checkpoint_steps)
    if alg == 'gd':
        train_(run_name=run_name, test_loader=testloader, train_loader=trainloader, alg=alg,
               num_training_samples=len(trainset),
               criterion=criterion_mean, test_criterion=criterion_sum,
               lr=lr, num_rounds=num_rounds, model_ori_path=ori_model,
               print_every=print_every, tb_every=tb_every, checkpoint_every=checkpoint_every, max_checked=max_test,
               log_args=args, log_argv=sys.argv,
               log_auc=log_auc,
               reserve_checkpoint_steps=reserve_checkpoint_steps)
    elif alg == 'fedavg':
        train_(run_name=run_name, test_loader=testloader, train_loader=trainloader, alg=alg,
               server=server, clients=clients, K=K, C=C, momentum=momentum, E=E, mu_FedProx=mu_FedProx,
               scale_in_it=scale_in_it,
               criterion=criterion_mean, test_criterion=criterion_sum,
               lr=lr, num_rounds=num_rounds, model_ori_path=ori_model,
               print_every=print_every, tb_every=tb_every, checkpoint_every=checkpoint_every,
               statistics_every=sta_every, max_checked=max_test,
               log_args=args, log_argv=sys.argv,
               log_auc=log_auc,
               reserve_checkpoint_steps=reserve_checkpoint_steps)
    elif alg == 'lastavg':
        train_(run_name=run_name, test_loader=testloader, train_loader=trainloader, alg=alg,
               server=server, clients=clients, K=K, C=C, E=E, num_non_update_rounds=num_non_update_rounds,
               scale_in_it=scale_in_it,
               criterion=criterion_mean, test_criterion=criterion_sum,
               lr=lr, num_rounds=num_rounds, model_ori_path=ori_model,
               print_every=print_every, tb_every=tb_every, checkpoint_every=checkpoint_every,
               statistics_every=sta_every, max_checked=max_test,
               log_args=args, log_argv=sys.argv,
               log_auc=log_auc,
               reserve_checkpoint_steps=reserve_checkpoint_steps)
    elif alg == 'waitavg':
        train_(run_name=run_name, test_loader=testloader, train_loader=trainloader, alg=alg,
               server=server, clients=clients, K=K, C=C, E=E, num_non_update_rounds=num_non_update_rounds,
               scale_in_it=scale_in_it,
               criterion=criterion_mean, test_criterion=criterion_sum,
               lr=lr, num_rounds=num_rounds, model_ori_path=ori_model,
               print_every=print_every, tb_every=tb_every, checkpoint_every=checkpoint_every,
               statistics_every=sta_every, max_checked=max_test,
               log_args=args, log_argv=sys.argv,
               log_auc=log_auc,
               reserve_checkpoint_steps=reserve_checkpoint_steps)


if __name__ == "__main__":
    main()
