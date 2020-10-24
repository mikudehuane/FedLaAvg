import os
import random

import _init_paths
import datetime
import pickle

import torch
from torch.utils.data import Dataset, sampler
from torch.nn.utils.rnn import pad_sequence
import common
import os.path as osp
import numpy as np

from train.client import Client
from utils import widgets, sentiment140
from utils.din import DeepInterestNetwork
from utils.sentiment140 import infer_sleep, time2int
from matplotlib import pyplot as plt
import utils.sentiment140

_WINDOW_SIZE = 14
_alibaba_train_fp = osp.join(common.alibaba_fd, "taobao_local_train")
_alibaba_train_fp_remap = osp.join(common.alibaba_fd, "taobao_local_train_remap")
_alibaba_test_fp = osp.join(common.alibaba_fd, 'taobao_local_test')
_alibaba_test_fp_remap = osp.join(common.alibaba_fd, "taobao_local_test_remap")
_split_column = ','
_split_his = ' '

# expose to other files
alibaba_train_fp = _alibaba_train_fp_remap
alibaba_test_fp = _alibaba_test_fp_remap


def str2datetime(dtime):
    return datetime.datetime(
        year=int(dtime[:4]),
        month=int(dtime[4:6]),
        day=int(dtime[6:8]),
        hour=int(dtime[8:10]),
        minute=int(dtime[11:13]),
        second=int(dtime[14:16])
    )


def _remap(src, dst):
    client_map = pickle.load(open(common.alibaba_uid_map_fp, 'rb'))
    good_map = pickle.load(open(common.alibaba_gid_map_fp, 'rb'))
    cat_map = pickle.load(open(common.alibaba_cid_map_fp, 'rb'))
    good_map['183215'] = 0  # this is the only good not in the map
    count_uid_notin = 0
    notin_clients = set()
    with open(src, 'r') as fin:
        with open(dst, 'w') as fout:
            for line_idx, line in enumerate(fin):
                line = line.strip().split('\t')
                click = line[AlibabaDataset.CLK]
                uid = line[AlibabaDataset.CLIENT]
                if uid not in client_map:
                    count_uid_notin += 1
                    notin_clients.add(uid)
                    uid = 0
                else:
                    uid = client_map[uid]
                gid = good_map[line[AlibabaDataset.GOOD]]
                cid = cat_map[line[AlibabaDataset.CAT]]
                gid_his = line[AlibabaDataset.GOOD_HIS]
                gid_his = ' '.join([str(good_map[_gid]) for _gid in gid_his.split('\x02')])
                cid_his = line[AlibabaDataset.CAT_HIS]
                cid_his = ' '.join([str(cat_map[_cid]) for _cid in cid_his.split('\x02')])
                fout.write(f"{click},{uid},{gid},{cid},{gid_his},{cid_his}\n")
                if line_idx % 100000 == 0:
                    print(f"{line_idx} lines processed", end='\r')
    print(f"{count_uid_notin} data in {src} with clients not in the map file, these are {len(notin_clients)} clients")


class AlibabaDatasetBasic(Dataset):
    def __init__(self, source_fp, max_length=100):
        """
        :param source_fp: source file path
        :param max_length: maximum length of user behvaior for training
        """
        super().__init__()
        self.data = open(source_fp).readlines()
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        datum = self.data[item]
        datum = datum.strip().split(_split_column)
        uid = int(datum[AlibabaDataset.CLIENT])
        gid = int(datum[AlibabaDataset.GOOD])
        cid = int(datum[AlibabaDataset.CAT])

        gid_his = datum[AlibabaDataset.GOOD_HIS]
        gid_his = [int(x) for x in gid_his.split(_split_his)]
        gid_his = torch.from_numpy(np.array(gid_his))
        cid_his = datum[AlibabaDataset.CAT_HIS]
        cid_his = [int(x) for x in cid_his.split(_split_his)]
        cid_his = torch.from_numpy(np.array(cid_his))

        if len(gid_his) > self.max_length:
            gid_his = gid_his[-self.max_length:]
            cid_his = cid_his[-self.max_length:]

        click = int(datum[AlibabaDataset.CLK])

        return uid, gid, cid, gid_his, cid_his, click


class AlibabaDataset(AlibabaDatasetBasic):
    # idx for the processed data
    CLK, CLIENT, GOOD, CAT, GOOD_HIS, CAT_HIS = range(6)

    def __init__(self, *args, **kwargs):
        """
        :param source_fp: source file path
        :param max_length: maximum length of user behvaior for training
        """
        super().__init__(*args, **kwargs)

        self.client_map = pickle.load(open(common.alibaba_uid_map_fp, 'rb'))
        self.good_map = pickle.load(open(common.alibaba_gid_map_fp, 'rb'))
        self.cat_map = pickle.load(open(common.alibaba_cid_map_fp, 'rb'))
        print(f"number of clients: {self.num_clients}, goods: {self.num_goods}, categories: {self.num_cats}")

    def shuffle_(self, seed=3):
        random.seed(3)
        random.shuffle(self.data)

    @property
    def num_clients(self):
        return len(self.client_map)

    @property
    def num_goods(self):
        return len(self.good_map)

    @property
    def num_cats(self):
        return len(self.cat_map)

    def partition_clients(self, **kwargs):
        filter_thres = kwargs.pop("filter_thres")
        filter_clients_up = kwargs.pop("filter_thres_up", 2**32)
        fd = osp.join(common.alibaba_fd, f"{filter_thres}_{filter_clients_up}")
        os.makedirs(fd, exist_ok=False)
        f = None
        last_client = None
        for idx, line in enumerate(self.data):
            datum = line.strip('\n').split(_split_column)
            client_id = datum[AlibabaDataset.CLIENT]
            if client_id != last_client:
                if f is not None:
                    f.close()
                f = open(osp.join(fd, client_id), 'a')
            last_client = client_id
            f.write(line)
        f.close()
        pickle.dump(dict(num_clients=self.num_clients), open(osp.join(fd, 'meta.pkl'), 'wb'))

    def filter_clients_(self, clients):
        """
        filter the dataset, reserve only data with client in clients
        """
        clients = set(client.id for client in clients)
        new_data = []
        for line in self.data:
            datum = line.strip().split(_split_column)
            if int(datum[AlibabaDataset.CLIENT]) in clients:
                new_data.append(line)
        self.data = new_data

    @staticmethod
    def collate_fn(batch):
        uids, gids, cids, gid_his, cid_his, clicks = tuple(zip(*batch))
        seq_lens = [len(h) for h in gid_his]

        uids = torch.from_numpy(np.array(uids, dtype=np.int64))
        gids = torch.from_numpy(np.array(gids, dtype=np.int64))
        cids = torch.from_numpy(np.array(cids, dtype=np.int64))
        clicks = torch.from_numpy(np.array(clicks, dtype=np.int64))

        gid_his = pad_sequence(gid_his, batch_first=True)
        cid_his = pad_sequence(cid_his, batch_first=True)
        masks = torch.zeros_like(gid_his, dtype=torch.float32)
        for idx, seq_len in enumerate(seq_lens):
            masks[idx, :seq_len] = 1.0
        return [uids, gids, cids, gid_his, cid_his, masks], clicks


def get_clients(data_fd, collate_fn=None, **kwargs):
    if collate_fn is None:
        collate_fn = AlibabaDataset.collate_fn
    num_clients = kwargs.pop("num_clients", -1)
    availability_file = kwargs.pop("availability_file", 'client_availability')
    random.seed(3)

    meta = pickle.load(open(osp.join(data_fd, 'meta.pkl'), 'rb'))
    client_available = pickle.load(open(osp.join(common.cache_fd, f'{availability_file}.pkl'), 'rb'))

    files = []
    for root, dirs, files in os.walk(data_fd):
        break
    # enforce same results each run
    files = sorted(files)

    clients = []
    if num_clients != -1:
        files = random.sample(files, num_clients)
    for fn in files:
        if fn != "meta.pkl":
            fp = osp.join(data_fd, fn)
            dataset = AlibabaDatasetBasic(fp)
            if isinstance(client_available, list):
                ar = client_available[int(fn)]
                client = Client(dataset, train_indices=None, is_available=("check_in_range", ar), collate_fn=collate_fn,
                                id=int(fn), **kwargs)
            elif isinstance(client_available, dict):
                ar = client_available['content'][int(fn)]
                client = Client(dataset, train_indices=None, is_available=("check_time", ar, client_available['num_hours1block']), collate_fn=collate_fn,
                                id=int(fn), **kwargs)
            clients.append(client)

    return clients


def _test():
    # _remap(_alibaba_test_fp, _alibaba_test_fp_remap)
    # _remap(_alibaba_train_fp, _alibaba_train_fp_remap)
    ali_trainset = AlibabaDataset(alibaba_train_fp)
    num_neg = 0
    num_posi = 0
    for idx, datum in enumerate(ali_trainset):
        click = datum[-1]
        if click == 1:
            num_posi += 1
        else:
            num_neg += 1
        if idx % 100000 == 0:
            print(idx)
    print(f"click {num_posi} ignore {num_neg}, ratio {num_posi / (num_posi + num_neg)}")
    # ali_testset = AlibabaDataset(alibaba_test_fp)
    # clients = get_clients(osp.join(common.alibaba_fd, "32_512"), num_clients=1000, availability_file="client_available_14400_86400_1.0")
    # widgets.plot_time2available_data_ratio(clients)
    # sentiment140.plot_availability_count(clients)

    # trainloader = torch.utils.data.DataLoader(ali_trainset, batch_size=1024, collate_fn=AlibabaDataset.collate_fn,
    #                                           shuffle=False)
    # count_batch = 0
    # import time
    # start = time.time()
    # while True:
    #     for x in trainloader:
    #         count_batch += 1
    #         if count_batch > 100:
    #             end = time.time()
    #             print(end - start)
    #             exit()
    # clients = ali_dataset.get_clients(train_batch_size=2)

    # ali_dataset.processed_data = ali_dataset.data   # to use sentiment140 functions
    # ali_dataset.SENTIMENT = ali_dataset.CLK
    # utils.sentiment140.plot_datum_time(ali_dataset, 'ali_time.png')
    # utils.sentiment140.plot_posi_ratio_with_time(ali_dataset)
    # utils.sentiment140.plot_availability_count(clients)
    # utils.sentiment140.plot_client_sample_volume(clients)


if __name__ == '__main__':
    _test()
