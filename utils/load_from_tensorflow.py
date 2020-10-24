import _init_paths
import pickle

import numpy as np
import torch

from train.model import DinDiceWhole
import os.path as osp

_model_fd = "/root/data/taobao_data_process/model_pars"


def load_embedding(fn, model_fd=_model_fd):
    def _init(m):
        fp = osp.join(model_fd, fn)
        weight = pickle.load(open(fp, 'rb'))
        m.weight.data[...] = torch.from_numpy(weight)
    return _init


def load_dice(fn):
    fp = osp.join(_model_fd, fn)
    weight = pickle.load(open(fp, 'rb'))
    return torch.from_numpy(weight)


def load_fc(weight_fn, bias_fn):
    def _init(m):
        weight_fp = osp.join(_model_fd, weight_fn)
        bias_fp = osp.join(_model_fd, bias_fn)
        weight = pickle.load(open(weight_fp, 'rb'))
        bias = pickle.load(open(bias_fp, 'rb'))
        m.weight.data[...] = torch.from_numpy(weight.transpose())
        m.bias.data[...] = torch.from_numpy(bias)
    return _init


def load_bn(gamma_fn, beta_fn):
    def _init(m):
        gamma_fp = osp.join(_model_fd, gamma_fn)
        beta_fp = osp.join(_model_fd, beta_fn)
        gamma = pickle.load(open(gamma_fp, 'rb'))
        beta = pickle.load(open(beta_fp, 'rb'))
        m.weight.data[...] = torch.from_numpy(gamma.transpose())
        m.bias.data[...] = torch.from_numpy(beta)
    return _init


net = DinDiceWhole()


def _test():
    net.din.user_embedding.apply(load_embedding("uid_embedding_var-0_49023-18.pkl"))
    net.din.cat_embedding.apply(load_embedding("cat_embedding_var-0_4815-18.pkl"))
    net.din.good_embedding.apply(load_embedding("mid_embedding_var-0_143534-18.pkl"))

    net.din.mlp.fc1.apply(load_fc("f1-kernel-0_162-200.pkl", "f1-bias-0_200.pkl"))
    net.din.mlp.fc2.apply(load_fc("f2-kernel-0_200-80.pkl", "f2-bias-0_80.pkl"))
    net.din.mlp.fc3.apply(load_fc("f3-kernel-0_80-2.pkl", "f3-bias-0_2.pkl"))
    net.din.mlp.bn.apply(load_bn("bn1-gamma-0_162.pkl", "bn1-beta-0_162.pkl"))
    net.din.mlp.a1.weight.data[...] = load_dice("dice_1-alphadice_1-0_200.pkl")
    net.din.mlp.a2.weight.data[...] = load_dice("dice_2-alphadice_2-0_80.pkl")

    net.din.attention.fc1.apply(load_fc("f1_att-kernel-0_144-80.pkl", "f1_att-bias-0_80.pkl"))
    net.din.attention.fc2.apply(load_fc("f2_att-kernel-0_80-40.pkl", "f2_att-bias-0_40.pkl"))
    net.din.attention.fc3.apply(load_fc("f3_att-kernel-0_40-1.pkl", "f3_att-bias-0_1.pkl"))

    from utils.alibaba import AlibabaDataset, alibaba_train_fp, alibaba_test_fp
    import utils.widgets as widgets
    # trainset = AlibabaDataset(alibaba_train_fp)
    testset = AlibabaDataset(alibaba_test_fp)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False,
                                             collate_fn=AlibabaDataset.collate_fn)
    test_acc_, _ = [-1.], [-1.]
    test_loss_, _ = [-1.], [-1.]
    test_auc_, _ = [-1.], [-1.]
    net.cuda()
    widgets.test_loader(net=net, loader=testloader, max_checked=1200000,
                        acc_=test_acc_, loss_=test_loss_, auc_=test_auc_, verbose=True)
    print("test_acc: %.3f, test_loss: %.3f, test_auc: %.3f" %
          (test_acc_[0], test_loss_[0], test_auc_[0]))

    debug = 1


if __name__ == "__main__":
    _test()