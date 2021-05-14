import os.path as osp
import os

import torch
from config import project_dir, use_cuda

cifar10_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

raw_data_dir = osp.join(project_dir, 'raw_data')
data_cache_dir = osp.join(project_dir, 'data')

cache_fd = osp.join(project_dir, 'cache')

sentiment140_fd = osp.join(raw_data_dir, 'Sentiment140')
sentiment140_fn = 'training.1600000.processed.noemoticon.csv'
sentiment140_pyobj_fn = 'training.1600000.processed.noemoticon.pkl'

alibaba_fd = osp.join(raw_data_dir, 'taobao_data_process', 'central_taobao_data')
alibaba_fp = osp.join(alibaba_fd, 'taobao-jointed-new')
alibaba_new_fp = osp.join(alibaba_fd, 'FedLaAvg_taobao-jointed-new_sorted.pkl')
alibaba_new_fp_csv = osp.join(alibaba_fd, 'FedLaAvg_taobao-jointed-new_remap.csv')
alibaba_uid_map_fp = osp.join(alibaba_fd, 'taobao_uid_voc.pkl')
alibaba_gid_map_fp = osp.join(alibaba_fd, 'taobao_mid_voc.pkl')
alibaba_cid_map_fp = osp.join(alibaba_fd, 'taobao_cat_voc.pkl')
alibaba_meta_fp = osp.join(alibaba_fd, 'taobao-item-info')

lr_configure_fd = osp.join(project_dir, 'train', 'lr_configures')

tb_slink_fd = osp.join(project_dir, 'log', "tensorboard")
record_fd = osp.join('log', "runs")
log_fd = "log"
checkpoint_fd = "checkpoint"
tensorboard_fd = "tensorboard"

figure_fd = osp.join(project_dir, "figure")
os.makedirs(figure_fd, exist_ok=True)

NUM_CIFAR10_CLASSES = 10
NUM_CIFAR10_TRAIN = 50000
NUM_CIFAR10_TEST = 10000

NUM_MNIST_CLASSES = 10
NUM_MNIST_TRAIN = 60000
NUM_MNIST_TEST = 10000

device = torch.device('cuda') if use_cuda else torch.device('cpu')

seed_for_train_test_partition = 1
seed_for_client_sampling = 1
seed_for_drop = 1
other_seed = 1

nlp_embedding_fd = osp.join(project_dir, 'models', 'glove')
