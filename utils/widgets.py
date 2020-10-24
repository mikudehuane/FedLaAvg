import datetime
import os
import pickle
import traceback
from copy import deepcopy

import torch
from torch import nn

import sklearn.metrics
import common
import numpy as np


def data_b_to_right_device(data_b):
    if isinstance(data_b, list):
        for idx in range(len(data_b)):
            data_b[idx] = data_b[idx].to(device=common.device)
    else:
        data_b = data_b.to(device=common.device)
    return data_b


def get_output(model, data_b):
    if isinstance(data_b, list):
        output_b = model(*data_b)
    else:
        output_b = model(data_b)
    return output_b


def get_spare_dir(f_dir, c_dir, new=True, cdir=False):
    """
    :param new: seek for new path
    :param f_dir: father directory
    :param c_dir: child directory prefix
    :return: a created idle directory like f_dir/c_dir0
    """
    test_num = 0
    while os.path.exists(os.path.join(f_dir, c_dir + str(test_num))):
        test_num += 1
    if not new:
        test_num -= 1
    assert test_num >= 0
    if cdir:
        return c_dir + str(test_num)
    else:
        idle_path = os.path.join(f_dir, c_dir + str(test_num))
        os.makedirs(idle_path, exist_ok=True)
        return idle_path


def init_pars(dst, src):
    """
    init self.pars in __init__
    :param dst: self.pars
    :param src: pars delivered
    :return: None
    """
    try:
        for k in src:
            if k not in dst:
                print(f"Invalid parameter {k}")
                raise AssertionError
            dst[k] = src[k]
    except AssertionError as e:
        traceback.print_exc()
        raise e


def test_loader(net: nn.Module, loader, max_checked=-1, criterion=nn.CrossEntropyLoss(reduction='sum'), average=True,
                loss_=None, acc_=None, grad_norm_=None, auc_=None, verbose=False):
    # use sum reduction for criterion please
    net = deepcopy(net)
    net.zero_grad()

    num_checked = 0
    num_right = 0
    total_loss = 0.0
    probs = []
    labels = []

    for idx, (data_b, label_b) in enumerate(loader):
        data_b = data_b_to_right_device(data_b)
        label_b = label_b.to(device=common.device)

        output_b = get_output(net, data_b)
        probs_ = torch.softmax(output_b, dim=1)
        probs_ = list(probs_.detach().cpu().numpy()[:, 1].reshape(-1))
        probs.extend(probs_)
        labels.extend(list(label_b.detach().cpu().numpy()))

        loss = criterion(output_b, label_b)
        loss.backward()
        with torch.no_grad():
            pred_b = torch.argmax(output_b, dim=1)
            num_right += (pred_b == label_b).sum().item()
        total_loss += loss.item()

        num_checked += len(label_b)
        if max_checked != -1 and num_checked >= max_checked:
            break

        if verbose and idx % 10 == 0:
            print(f"{idx}/{len(loader)} batches tested", end='\r')
    grad_norm_squ = 0.0
    for par in net.parameters():
        grad_norm_squ += torch.norm(par.grad) ** 2
    grad_norm = grad_norm_squ ** 0.5

    if average:
        num_right /= num_checked
        total_loss /= num_checked
        grad_norm /= num_checked

    if loss_ is not None:
        loss_[0] = total_loss
    if acc_ is not None:
        acc_[0] = num_right
    if grad_norm_ is not None:
        grad_norm_[0] = grad_norm
    if auc_ is not None:
        try:
            auc_[0] = sklearn.metrics.roc_auc_score(labels, probs)
        except ValueError as e:
            print(f"probs: {probs}")
            raise e


def latest_gradient_avg(clients):
    if len(clients) == 0:
        raise ValueError("no client given")
    lg_sum = deepcopy(clients[0].latest_grad)
    for client in clients[1:]:
        lg = client.latest_grad
        try:
            lg_sum = [lg_ + lgs_ for lg_, lgs_ in zip(lg, lg_sum)]
        except TypeError as e:
            if lg is None or lg_sum is None:
                # lg_sum remain as itself
                lg_sum = lg_sum
            else:
                print(f"lg is {type(lg)}, lg_sum is {type(lg_sum)}")
                raise e
    if lg_sum is None:
        return None
    else:
        return [lgs / len(clients) for lgs in lg_sum]


def rel_and_norm_error(tensor_list1, tensor_list2, eps=1e-7):
    # tensor_list is None means that it is zero
    if tensor_list1 is None and tensor_list2 is None:
        return 0.0, 0.0
    else:
        if tensor_list1 is None:
            tensor_list1 = [torch.zeros_like(t) for t in tensor_list2]
        elif tensor_list2 is None:
            tensor_list2 = [torch.zeros_like(t) for t in tensor_list1]
    tensor_list1 = [t1.to(device=t2.device) for t1, t2 in zip(tensor_list1, tensor_list2)]
    diff = [t1 - t2 for t1, t2 in zip(tensor_list1, tensor_list2)]
    diff_norm = 0.0
    for diff_ in diff:
        diff_norm += (torch.norm(diff_) ** 2).item()
    diff_norm = diff_norm ** 0.5
    rel_errors = [torch.abs(d) / (torch.abs(t1) + torch.abs(t2) + eps)
                  for d, t1, t2 in zip(diff, tensor_list1, tensor_list2)]
    rel_errors = [torch.max(rel).item() for rel in rel_errors]
    return max(rel_errors), diff_norm


def count_parameters(model, verbose=False):
    count = 0
    for par in model.parameters():
        _count = np.prod(par.shape)
        count += _count
    return count


def _check_local_test_or_train(fp, user_behaviors):
    """
    check:
    log time are Monotonically increasing
    """
    with open(fp, 'r') as f:
        for line_idx, line in enumerate(f):
            line = line.strip('\n')
            click, uid, gid, cid, gid_his, cid_his = line.split('\t')

            # gid_his = gid_his.split('\x02')
            # if uid not in user_behaviors:
            #     user_behaviors[uid] = gid_his
            # if click == "1":
            #     user_behaviors[uid].append(gid)

            if _user_behaviors_load[uid].find(gid_his) == -1:
                print(f"line {line_idx}")
                print(f"gid_his = {gid_his}")
                print(f"not in")
                print(f"user_behavior = {_user_behaviors_load[uid]}")
                input("continue?")

            if line_idx % 1000000 == 0:
                print(line_idx)


if __name__ == '__main__':
    import os.path as osp
    fps = osp.join(common.alibaba_fd, 'taobao_local_train'), osp.join(common.alibaba_fd, 'taobao_local_test')
    _user_behaviors_load = pickle.load(open(osp.join(common.cache_fd, 'user_behaviors_in_taobao_local_xxx.pkl'), 'rb'))
    _user_behaviors_load = {key: '\x02'.join(val) for key, val in _user_behaviors_load.items()}
    _user_behaviors = dict()
    for _fp in fps:
        _check_local_test_or_train(_fp, _user_behaviors)
    # pickle.dump(_user_behaviors, open(osp.join(common.cache_fd, 'user_behaviors_in_taobao_local_xxx.pkl'), 'wb'))


def plot_time2available_data_ratio(clients, fn_abs="time2available_data_abs.png", fn_ratio="time2available_data_ratio.png"):
    from matplotlib import pyplot as plt
    import os.path as osp
    posi_counts = [0 for hour in range(24)]
    total_counts = [0 for hour in range(24)]
    client_sample_count = [0 for client in clients]
    client_posi_count = [0 for client in clients]
    for idx, client in enumerate(clients):
        client_sample_count[idx] = len(client.dataset)
        for uid, gid, cid, gid_his, cid_his, click in client.dataset:
            if click == 1:
                client_posi_count[idx] += 1
    for hour in range(24):
        time_ = datetime.timedelta(hours=hour)
        avai_count = 0
        for idx, client in enumerate(clients):
            if client.is_available(time_):
                posi_counts[hour] += client_posi_count[idx]
                total_counts[hour] += client_sample_count[idx]
    plt.figure(0)
    plt.plot(range(24), posi_counts)
    plt.plot(range(24), total_counts)
    plt.savefig(osp.join(common.figure_fd, fn_abs))
    plt.close('all')
    plt.figure(0)
    plt.plot(range(24), np.array(posi_counts) / np.array(total_counts))
    plt.savefig(osp.join(common.figure_fd, fn_ratio))


# def test_clients(net, clients, criterion):
#     # use sum reduction for criterion please
#     num_checked = 0
#     num_right = 0
#     total_loss = 0.0
#     for client in clients:
#         _nrt, _loss, _nck = \
#             test_loader(net, client.dataloader, criterion=criterion, average=False)
#         num_checked += _nck
#         num_right += _nrt
#         total_loss += _loss
#     train_acc = num_right / num_checked
#     train_loss = total_loss / num_checked
#     return train_acc, train_loss
