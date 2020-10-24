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
from utils.alibaba import get_clients
from utils.din import DeepInterestNetwork
from utils.sentiment140 import infer_sleep, time2int
from matplotlib import pyplot as plt
import utils.sentiment140

_WINDOW_SIZE = 14


def str2datetime(dtime):
    return datetime.datetime(
        year=int(dtime[:4]),
        month=int(dtime[4:6]),
        day=int(dtime[6:8]),
        hour=int(dtime[8:10]),
        minute=int(dtime[11:13]),
        second=int(dtime[14:16])
    )


class AlibabaSourceDatasetForInferAvailability(Dataset):
    # idx for the processed data
    DATETIME, CLK, CLIENT, GOOD, CATEGORY = range(5)

    def __init__(self):
        super().__init__()

        # remap dataset
        # preprocess DATETIME as datetime, others as int
        # sort by datetime
        self.data, self.num_clients, self.num_goods, self.num_cats = self.__get_preprocessed_data()
        print(f"number of clients: {self.num_clients}, goods: {self.num_goods}, categories: {self.num_cats}")

        print(f"Data collected "
              f"from {self.data[0][AlibabaSourceDatasetForInferAvailability.DATETIME].isoformat(sep=' ')} "
              f"to {self.data[-1][AlibabaSourceDatasetForInferAvailability.DATETIME].isoformat(sep=' ')}")

        # get the first data index in the 15th day
        # first_date: the date of the first datum
        self.first_date = self.data[0][AlibabaSourceDatasetForInferAvailability.DATETIME].date()
        self.last_date = self.data[-1][AlibabaSourceDatasetForInferAvailability.DATETIME].date()
        self.train_start = self.get_data_start(num_days=_WINDOW_SIZE, from_=14005500)  # 15005575
        self.test_start = self.get_data_start(num_days=(self.last_date - self.first_date).days, from_=28000000)  # 31312950
        print(f"training data from {self.train_start}, test data from {self.test_start}, total data {len(self.data)}")

        # get the clients
        self.client_behaviors = self.get_client_behaviors()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        datum = self.data[item]
        uid = datum[AlibabaSourceDatasetForInferAvailability.CLIENT]
        gid = datum[AlibabaSourceDatasetForInferAvailability.GOOD]
        cid = datum[AlibabaSourceDatasetForInferAvailability.CATEGORY]

        client_behavior = self.client_behaviors[uid]
        end_day = (datum[AlibabaSourceDatasetForInferAvailability.DATETIME].date() - self.first_date).days  # not included
        start_day = end_day - _WINDOW_SIZE
        gid_his = torch.from_numpy(np.array([gid for day, gid, cid in client_behavior if start_day <= day < end_day],
                                            dtype=np.int64))
        cid_his = torch.from_numpy(np.array([cid for day, gid, cid in client_behavior if start_day <= day < end_day],
                                            dtype=np.int64))

        click = datum[AlibabaSourceDatasetForInferAvailability.CLK]

        return uid, gid, cid, gid_his, cid_his, click

    def get_clients_available(self, ratio2availability=lambda x: 8*3600):
        """
        :param ratio2availability: a function, input is positive click ratio, output is number of *seconds* to be available
        :return: an array of client sleeping time (start, end), unit: seconds, array indexed by client id (remaped)
        """
        clients_click_seconds = [[] for _ in range(self.num_clients)]

        id2posi = [0 for _ in range(self.num_clients)]
        id2total = [0 for _ in range(self.num_clients)]
        for idx, datum in enumerate(self.data):
            client_id = datum[AlibabaSourceDatasetForInferAvailability.CLIENT]
            click_time = datum[AlibabaSourceDatasetForInferAvailability.DATETIME]
            click_time = time2int(click_time)
            clients_click_seconds[client_id].append(click_time)

            id2total[client_id] += 1
            if datum[AlibabaSourceDatasetForInferAvailability.CLK] == 1:
                id2posi[client_id] += 1
        id2ratio = [posi / (total + 1e-7) for posi, total in zip(id2posi, id2total)]
        # print(id2ratio)
        plt.figure(0)
        for idx, ratio in enumerate(id2ratio):
            plt.scatter(random.random(), ratio)
            if idx >= 1000:
                break
        plt.savefig(osp.join(common.figure_fd, 'client_positive_ratio.png'))
        plt.close('all')

        num_seconds_day = 24 * 3600
        if isinstance(ratio2availability, tuple):
            if ratio2availability[0] == 'uniform_map':
                function, range_min, range_max, max_clip = ratio2availability
                ratio_min = min(id2ratio)
                ratio_max = min(max(id2ratio), max_clip)
                print(f"ration_min={ratio_min}, ratio_max={ratio_max}")

                def ratio2availability(_ratio):
                    return (min(_ratio, max_clip) - ratio_min) * (range_max - range_min) / (ratio_max - ratio_min) + range_min
            else:
                raise ValueError(f"Unaccepted {ratio2availability}")
        client_available = [None for _ in range(self.num_clients)]
        for client_id, click_time in enumerate(clients_click_seconds):
            posi_ratio = id2ratio[client_id]
            num_seconds_sleep = ratio2availability(posi_ratio)
            if len(click_time) != 0:
                sleep_range = infer_sleep(click_time, num_seconds_day=num_seconds_day, num_seconds_sleep=num_seconds_sleep)
            else:
                sleep_range = (0, num_seconds_sleep)
            client_available[client_id] = sleep_range
        return client_available

    def get_clients_available_block(self, num_hours1block, include_neg=False, thres=1, reverse=False):
        random.seed(1)
        clients_available_block = [[] for _ in range(self.num_clients)]
        client_block2count = dict()
        for idx, datum in enumerate(self.data):
            if datum[AlibabaSourceDatasetForInferAvailability.CLK] == 1 or include_neg:
                client_id = datum[AlibabaSourceDatasetForInferAvailability.CLIENT]
                click_time = datum[AlibabaSourceDatasetForInferAvailability.DATETIME]
                click_time = time2int(click_time)
                click_hour = click_time // 3600
                click_block = click_hour // num_hours1block
                if (client_id, click_block) not in client_block2count:
                    client_block2count[(client_id, click_block)] = 0
                client_block2count[(client_id, click_block)] += 1
        if not reverse:
            for client_id, click_block in client_block2count:
                if client_block2count[(client_id, click_block)] >= thres:
                    clients_available_block[client_id].append(click_block)
        else:
            for client_id in range(self.num_clients):
                for click_block in range(24//num_hours1block):
                    if client_block2count.get((client_id, click_block), 0) <= thres:
                        clients_available_block[client_id].append(click_block)
        for idx, ab in enumerate(clients_available_block):
            if len(ab) == 0:
                click_time = random.randint(0, 3600*24-1)
                click_hour = click_time // 3600
                client_block = click_hour // num_hours1block
                clients_available_block[idx].append(client_block)
        clients_available_block = [set(x) for x in clients_available_block]
        return dict(content=clients_available_block, type='block', num_hours1block=num_hours1block)

    @staticmethod
    def __get_preprocessed_data(map_="load"):
        if not osp.exists(common.alibaba_new_fp):
            c_client = 0
            c_good = 0
            c_cat = 0
            if map_ == "generate":
                client_map = dict()
                good_map = dict()
                cat_map = dict()
            elif map_ == "load":
                client_map = pickle.load(open(common.alibaba_uid_map_fp, 'rb'))
                good_map = pickle.load(open(common.alibaba_gid_map_fp, 'rb'))
                cat_map = pickle.load(open(common.alibaba_cid_map_fp, 'rb'))
            else:
                raise ValueError(f"unrecognized map_={map_}")

            # get the original gid-cid map
            good2cat_original = dict()
            for line in open(common.alibaba_meta_fp, 'r'):
                line = line.strip('\n').split('\t')
                gid, cid = line
                gid, cid = int(gid), int(cid)
                if gid not in good2cat_original:
                    good2cat_original[gid] = cid
                else:
                    if good2cat_original[gid] != cid:
                        print(f"the good {gid} has multiple categories ({cid} and {good2cat_original[gid]})")

            processed_data = []
            with open(common.alibaba_fp, 'r') as f_in:
                with open(common.alibaba_new_fp_csv, 'w') as f_out:
                    for line_idx, line in enumerate(f_in):
                        line = line.strip('\n').split('\t')
                        dtime = str2datetime(line[AlibabaSourceDatasetForInferAvailability.DATETIME])
                        click = int(line[AlibabaSourceDatasetForInferAvailability.CLK])
                        client_id = line[AlibabaSourceDatasetForInferAvailability.CLIENT]
                        good_id = line[AlibabaSourceDatasetForInferAvailability.GOOD]
                        cat_id = line[AlibabaSourceDatasetForInferAvailability.CATEGORY]

                        # check meta
                        if good2cat_original[int(good_id)] != int(cat_id):
                            print(f"good {good_id} is mapped to category {good2cat_original[int(good_id)]} in the meta"
                                  f" but recorded as {cat_id} in line {line_idx}")

                        if map_ == 'generate':
                            if client_id not in client_map:
                                client_map[client_id] = c_client
                                c_client += 1
                            if good_id not in good_map:
                                good_map[good_id] = c_good
                                c_good += 1
                            if cat_id not in cat_map:
                                cat_map[cat_id] = c_cat
                                c_cat += 1
                        if map_ == 'load':
                            if good_id not in good_map:
                                print(f"{good_id} not in good_map", end='')
                                if client_id not in client_map:
                                    print(f", the corresponding client {client_id} also not", end="")
                                print()
                                c_good += 1
                            if cat_id not in cat_map:
                                print(f"{cat_id} not in cat_map")
                                c_cat += 1
                            if client_id not in client_map:
                                c_client += 1
                                # ignore this datum if the client is not in the map
                                continue
                        client_id = client_map[client_id]
                        good_id = good_map.get(good_id, 0)
                        cat_id = cat_map.get(cat_id, 0)

                        processed_data.append([dtime, click, client_id, good_id, cat_id])
                        f_out.write(','.join([dtime.isoformat(sep=' '), str(click), str(client_id), str(good_id), str(cat_id)]) + '\n')
                        if line_idx % 10000 == 0:
                            print(f"{line_idx}/32345219 lines processed", end='\r')
            if map_ == 'load':
                print(f"count not in voc")
                print(f"number of clients: {c_client}, goods: {c_good}, categories: {c_cat}")
            print("data preprocessed")
            processed_data = sorted(processed_data, key=lambda x: x[AlibabaSourceDatasetForInferAvailability.DATETIME])
            num_clients = len(client_map)
            num_goods = len(good_map)
            num_cats = len(cat_map)
            pickle.dump((processed_data, num_clients, num_goods, num_cats),
                        open(common.alibaba_new_fp, 'wb'))
            data = processed_data
        else:
            print("loading preprocessed data file...")
            data, num_clients, num_goods, num_cats = pickle.load(open(common.alibaba_new_fp, 'rb'))
            print("loaded")
        return data, num_clients, num_goods, num_cats

    def get_data_start(self, num_days, from_):
        first_datetime = self.data[0][AlibabaSourceDatasetForInferAvailability.DATETIME]
        first_date = first_datetime.date()
        for idx, datum in enumerate(self.data[from_:], start=from_):
            if (datum[AlibabaSourceDatasetForInferAvailability.DATETIME].date() - first_date).days >= num_days:
                if idx == from_:
                    raise RuntimeError("A wrong data_start may be generated because of too large from_")
                else:
                    return idx

    def get_client_behaviors(self):
        client_behaviors = [[] for _ in range(self.num_clients)]
        for datum in self.data:
            if datum[AlibabaSourceDatasetForInferAvailability.CLK] == 1:
                client_id = datum[AlibabaSourceDatasetForInferAvailability.CLIENT]
                good_id = datum[AlibabaSourceDatasetForInferAvailability.GOOD]
                cat_id = datum[AlibabaSourceDatasetForInferAvailability.CATEGORY]
                days = (datum[AlibabaSourceDatasetForInferAvailability.DATETIME].date() - self.first_date).days
                # noinspection PyTypeChecker
                client_behaviors[client_id].append((days, good_id, cat_id))
        return client_behaviors


def _test():
    ali_dataset = AlibabaSourceDatasetForInferAvailability()

    # available_min, available_max = 4*3600, 24*3600
    # max_clip = 1.0
    # client_available = ali_dataset.get_clients_available(ratio2availability=('uniform_map', available_min, available_max, max_clip))

    num_blocks = 6
    include_neg = True
    thres = 16
    reverse = True
    client_available = ali_dataset.get_clients_available_block(24//num_blocks, include_neg=include_neg, thres=thres, reverse=reverse)
    plt.figure(1)
    for idx, ab in enumerate(client_available['content']):
        duration = len(ab)
        plt.scatter(random.random(), duration)
        if idx >= 1000:
            break
    plt.savefig(osp.join(common.figure_fd, "availability_duration.png"))
    plt.close('all')

    availability_file = f"client_available_nb{num_blocks}_in{include_neg}_th{thres}_r{reverse}.pkl"
    pickle.dump(client_available, open(osp.join(common.cache_fd, availability_file), 'wb'))

    clients = get_clients(osp.join(common.alibaba_fd, "32_512"), num_clients=1000,
                          availability_file=availability_file[:-4])
    sentiment140.plot_availability_count(clients)
    widgets.plot_time2available_data_ratio(clients, fn_abs=f"time2available_data_abs_{availability_file}.png",
                                           fn_ratio=f"time2available_data_ratio_{availability_file}.png")


if __name__ == '__main__':
    _test()
