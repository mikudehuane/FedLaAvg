import collections
import copy
import math
import random

# import _init_paths
import pickle
import re

import torch
from torch.nn.utils.rnn import pack_sequence

import common
from torch.utils.data import Dataset
import os.path as osp
import numpy as np
import datetime
import string
from bs4 import BeautifulSoup

from matplotlib import pyplot as plt

from train.client import Client

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

from train.model import DCNN, LSTM2

class Sentiment140Dataset(Dataset):
    SENTIMENT, ID, DATETIME, QUERY, CLIENT_ID, TWEET = range(6)

    def __init__(self, preprocess, embedding_fn=None, transform=None):
        super().__init__()
        if transform == 'glove_trans':
            self.transform = self.glove_trans
        else:
            self.transform = transform

        dataset_id = preprocess.id

        # process pars
        # get cache fp
        cache_fn = dataset_id + '.pkl'
        self.cache_fp = osp.join(common.data_cache_dir, cache_fn)

        # load and process data
        raw_data_dump_fp = osp.join(common.sentiment140_fd, common.sentiment140_pyobj_fn)
        if osp.exists(raw_data_dump_fp):
            raw_data = pickle.load(open(raw_data_dump_fp, 'rb'))
            print(f"raw data loaded from {raw_data_dump_fp}")
        else:
            raw_data = []
            with open(osp.join(common.sentiment140_fd, common.sentiment140_fn), 'r', encoding='latin1') as f:
                for line_idx, line in enumerate(f.readlines()):
                    tup = line.split('","')
                    # remove ""
                    tup[0] = tup[0][1:]
                    tup[-1] = tup[-1][:-1]
                    raw_data.append(tup)
                    if line_idx > 0:
                        assert len(raw_data[-1]) == len(raw_data[-2])
            pickle.dump(raw_data, open(raw_data_dump_fp, 'wb'))
        self.raw_data = raw_data

        print(f"processed data cached file path: \"{self.cache_fp}\"")
        if osp.exists(self.cache_fp):
            self.processed_data = pickle.load(open(self.cache_fp, 'rb'))
            print(f"processed data loaded from {self.cache_fp}")
        else:
            self.processed_data = preprocess(self.raw_data)
            pickle.dump(self.processed_data, open(self.cache_fp, 'wb'))
            print(f"processed data dumped into {self.cache_fp}")

        if embedding_fn is not None:
            if embedding_fn.split('.')[-1] not in ('pkl', 'txt'):
                embedding_fn += '.pkl'
            needed_glove_fn = 'sentiment140_' + '.'.join(embedding_fn.split('.')[:-1]) + '.pkl'
            needed_glove_fp = osp.join(common.cache_fd, needed_glove_fn)
            if osp.exists(needed_glove_fp):
                self.glove_model = pickle.load(open(needed_glove_fp, 'rb'))
                print(f"needed glove model loaded from {needed_glove_fp}")
            else:
                embedding_fp = osp.join(common.nlp_embedding_fd, embedding_fn)
                embedding_ext = embedding_fn.split('.')[-2:]
                if embedding_ext[-1] == 'pkl':
                    glove_model = pickle.load(open(embedding_fp, 'rb'))
                    print(f"whole glove model loaded from {embedding_fp}")
                elif embedding_ext[-1] == 'txt':
                    if embedding_ext[0] != 'word2vec':
                        new_embedding_fn = '.'.join(embedding_fn.split('.')[:-1]) + '.word2vec.txt'
                        new_embedding_fp = osp.join(common.nlp_embedding_fd, new_embedding_fn)
                        glove2word2vec(embedding_fp, new_embedding_fp)
                        embedding_fp = new_embedding_fp
                        print("glove format to word2vec done")
                    glove_model = KeyedVectors.load_word2vec_format(embedding_fp, binary=False)
                    pkl_fn = '.'.join(embedding_fn.split('.')[:-1]) + '.pkl'
                    pkl_fp = osp.join(common.nlp_embedding_fd, pkl_fn)
                    pickle.dump(glove_model, open(pkl_fp, 'wb'))
                    print(f"glove model dumped into {pkl_fp}")
                else:
                    raise ValueError(f"unrecognized embedding file {embedding_fp}")
                needed_glove_model = dict()
                for data_tup in self.processed_data:
                    tweet = data_tup[Sentiment140Dataset.TWEET]
                    for word in tweet.split():
                        if word in glove_model:
                            needed_glove_model[word] = glove_model[word]
                intermediate_fp = osp.join(common.cache_fd, needed_glove_fn + '.tmp.txt')
                with open(intermediate_fp, 'w') as f:
                    _w, _v = iter(needed_glove_model.items()).__next__()
                    f.write("{0} {1}\n".format(len(needed_glove_model), len(_v)))
                    for word, vec in needed_glove_model.items():
                        vec = [str(ele) for ele in vec]
                        vec = ' '.join([word] + vec)
                        f.write(vec + '\n')
                self.glove_model = KeyedVectors.load_word2vec_format(intermediate_fp, binary=False)
                pickle.dump(self.glove_model, open(needed_glove_fp, 'wb'))
                print(f"needed glove model dumped into {needed_glove_fp}")
        else:
            self.glove_model = None

    def statistic_words(self, verbose=True, num_print=1024):
        words2num = {}
        summ = 0
        for idx, data_tup in enumerate(self.processed_data):
            if verbose and idx % 1000 == 0:
                print(f"{idx}/{len(self.processed_data)}", end='\r')
            tweet = data_tup[Sentiment140Dataset.TWEET]
            words = tweet.split()
            for word in words:
                if word in words2num:
                    words2num[word] += 1
                else:
                    words2num[word] = 1
                summ += 1
        words2num = list(words2num.items())
        words2num = sorted(words2num, key=lambda pair: -pair[1])
        plot_data_abs = []
        plot_data_accumulate = []
        accumulate = 0
        for idx, (word, count) in enumerate(words2num):
            plot_data_abs.append(count)
            accumulate += count
            plot_data_accumulate.append(accumulate)
            if verbose and idx >= len(words2num) - num_print:
                print(f"{word}: {count}, total: {accumulate}/{summ}")
        plt.figure(0)
        plt.plot(list(range(len(words2num))), plot_data_abs)
        plt.savefig(osp.join(common.figure_fd, "words_distribution.png"))
        plt.close(0)
        plt.figure(1)
        plt.plot(list(range(len(words2num))), plot_data_accumulate)
        plt.savefig(osp.join(common.figure_fd, "words_pdf.png"))
        plt.close(1)

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, item):
        tup = self.processed_data[item]
        tweet = tup[Sentiment140Dataset.TWEET]
        label = tup[Sentiment140Dataset.SENTIMENT]
        if self.transform is not None:
            tweet = self.transform(tweet)
        return tweet, label

    def count_clients(self):
        client_set = set()
        for data_tup in self.processed_data:
            client_set.add(data_tup[Sentiment140Dataset.CLIENT_ID])
        return len(client_set)

    def count_words(self):
        words_set = set()
        for idx, data_tup in enumerate(self.processed_data):
            tweet = data_tup[Sentiment140Dataset.TWEET]
            words = tweet.split()
            for word in words:
                words_set.add(word)
        return len(words_set)

    def count_chars(self):
        ch_set = set()

        for data_tup in self.processed_data:
            tweet = data_tup[Sentiment140Dataset.TWEET]
            for ch in tweet:
                if ch not in ch_set:
                    # print(ch)
                    ch_set.add(ch)
        return len(ch_set)

    def partition(self, test_ratio=0.1, seed=common.seed_for_train_test_partition):
        # share glove_model
        random.seed(seed)
        num_total = len(self.processed_data)
        num_test = int(num_total * test_ratio)
        test_indices = random.sample(list(range(num_total)), num_test)
        test_indicators = [False] * num_total
        for idx in test_indices:
            test_indicators[idx] = True

        raw_data = self.raw_data
        processed_data = self.processed_data
        glove_model = self.glove_model
        self.raw_data = []
        self.processed_data = []
        self.glove_model = None
        testset = copy.deepcopy(self)
        trainset = copy.deepcopy(self)
        self.glove_model = glove_model
        trainset.glove_model = glove_model
        testset.glove_model = glove_model

        for idx, is_test in enumerate(test_indicators):
            if is_test:
                testset.raw_data.append(raw_data[idx])
                testset.processed_data.append(processed_data[idx])
            else:
                trainset.raw_data.append(raw_data[idx])
                trainset.processed_data.append(processed_data[idx])

        self.raw_data = raw_data
        self.processed_data = processed_data
        return trainset, testset

    def filter_clients_(self, threshold, up=2**32):
        clients = self.get_clients(train_batch_size=1)
        indices = []
        for client in clients:
            if up >= len(client.train_indices) > threshold:
                indices.extend(client.train_indices)
        self.raw_data = [self.raw_data[idx] for idx in indices]
        self.processed_data = [self.processed_data[idx] for idx in indices]

    def random_select_clients_(self, N, seed=common.seed_for_client_sampling):
        random.seed(seed)
        clients = self.get_clients(train_batch_size=1)
        use_clients = random.sample(clients, N)
        indices = []
        for client in use_clients:
            indices.extend(client.train_indices)
        self.raw_data = [self.raw_data[idx] for idx in indices]
        self.processed_data = [self.processed_data[idx] for idx in indices]

    def get_clients(self, collate_fn=None, strategy="blocked", **kwargs):
        """
        :param strategy: chosen from (blocked, modeled_mid)
            blocked: when data available in some block, always available in this block
            modeled_mid: get the middle time of all tweets, num_hours_busy hours around this time are unavailable
        :param collate_fn: passed to client.dataloader
        :param kwargs: customized parameters for each strategy
        :return: clients
        """
        if strategy not in ("blocked", "modeled_mid"):
            raise ValueError(f"strategy={strategy} not recognized")
        if collate_fn is None:
            collate_fn = Sentiment140Dataset.collate_fn
        num_hours1block = kwargs.pop("num_hours1block", 1)
        num_hours_busy = kwargs.pop("num_hours_busy", 16)

        client2id = {}
        clients = []  # index: id from 0, content: data indices
        clients_available_data = []

        current_id = 0
        for idx, data_tup in enumerate(self.processed_data):
            client_name = data_tup[Sentiment140Dataset.CLIENT_ID]
            if client_name in client2id:
                client_id = client2id[client_name]
            else:
                client2id[client_name] = current_id
                client_id = current_id
                clients.append([])
                if strategy == "blocked":
                    clients_available_data.append(set())
                elif strategy == "modeled_mid":
                    clients_available_data.append([])
                else:
                    raise ValueError(f"strategy={strategy} not recognized")
                current_id += 1
            clients[client_id].append(idx)
            if strategy == "blocked":
                tweet_hour = data_tup[Sentiment140Dataset.DATETIME].hour
                tweet_block = tweet_hour // num_hours1block
                clients_available_data[client_id].add(tweet_block)
            elif strategy == "modeled_mid":
                tweet_time = data_tup[Sentiment140Dataset.DATETIME]
                tweet_time_int = time2int(tweet_time)
                clients_available_data[client_id].append(tweet_time_int)
            else:
                raise ValueError(f"strategy={strategy} not recognized")

        if strategy == "blocked":
            clients = [Client(self, indices, ("check_time", a_blocks, num_hours1block),
                              id=client_id, collate_fn=collate_fn, **kwargs)
                       for client_id, (indices, a_blocks) in enumerate(zip(clients, clients_available_data))]
        elif strategy == "modeled_mid":
            num_seconds_day = 24 * 3600
            num_seconds_sleep = num_seconds_day - num_hours_busy * 3600
            clients = [Client(self, indices,
                              ("check_in_range", infer_sleep(tweets_time, num_seconds_day=num_seconds_day, num_seconds_sleep=num_seconds_sleep)),
                              id=client_id, collate_fn=collate_fn, **kwargs)
                       for client_id, (indices, tweets_time) in enumerate(zip(clients, clients_available_data))]
        else:
            raise ValueError(f"strategy={strategy} not recognized")
        return clients

    def glove_trans(self, tweet):
        words = tweet.split()
        word_vecs = []
        for word in words:
            try:
                word_vecs.append(self.glove_model[word])
            except KeyError:
                word_vecs.append(np.zeros(self.glove_model.vector_size, dtype=self.glove_model.vectors.dtype))
        return np.stack(word_vecs)

    def count_in_gloves(self):
        words_set = set()
        num_in = 0
        for idx, data_tup in enumerate(self.processed_data):
            tweet = data_tup[Sentiment140Dataset.TWEET]
            words = tweet.split()
            for word in words:
                if word not in words_set:
                    words_set.add(word)
                    if word in self.glove_model:
                        num_in += 1
                    else:
                        pass
                        # print(word)
        print(f"{num_in}/{len(words_set)} words in glove_model")

    def time_variation_(self, num_blocks=6, max_prop=1.0, min_prop=0.0, seed=common.seed_for_drop):
        """
        drop samples to enforce distribution variation with time
        max_prop, min_prop: maximum or minimum proportion of positive sentiment
        """
        random.seed(seed)
        if 24 % num_blocks != 0 or num_blocks % 2 != 0:
            raise ValueError(f"num_blocks={num_blocks} should be a even number that can divide 24")
        num_hours1block = 24 // num_blocks
        # partition data by time
        blocks_indices = [[[], []] for _ in range(num_blocks)]
        for data_idx, data_tup in enumerate(self.processed_data):
            block_idx = data_tup[Sentiment140Dataset.DATETIME].hour // num_hours1block
            sentiment = data_tup[Sentiment140Dataset.SENTIMENT]
            # noinspection PyTypeChecker
            blocks_indices[block_idx][sentiment].append(data_idx)
        # drop data
        indices = []
        for block_idx, block_indices in enumerate(blocks_indices):
            num_posi = len(block_indices[1])
            num_nega = len(block_indices[0])
            num_total = num_posi + num_nega
            num_blocks_mid = num_blocks // 2
            block_idx_mid = block_idx if block_idx <= num_blocks_mid else num_blocks - block_idx
            target_posi_ratio = (max_prop - min_prop) * (block_idx_mid / num_blocks_mid) + min_prop
            posi_ratio = num_posi / num_total
            # 1 for drop posi; 0 for drop nega
            if posi_ratio > target_posi_ratio:
                drop_group = 1
                drop_remain_ratio = target_posi_ratio / (1 - target_posi_ratio)
            else:
                drop_group = 0
                drop_remain_ratio = (1 - target_posi_ratio) / target_posi_ratio
            remain_group = 1 - drop_group
            num_remain_group = len(block_indices[remain_group])
            num_drop_remain = math.floor(num_remain_group * drop_remain_ratio)
            block_indices[drop_group] = random.sample(block_indices[drop_group], num_drop_remain)
            indices.extend(block_indices[0])
            indices.extend(block_indices[1])

        # filter data
        self.raw_data = [self.raw_data[idx] for idx in indices]
        self.processed_data = [self.processed_data[idx] for idx in indices]

    @staticmethod
    def collate_fn(batch):
        tweets, labels = tuple(zip(*batch))
        seq_lens = [len(tweet) for tweet in tweets]
        order = np.argsort(seq_lens)[::-1]
        # TODO(islander): variant length
        tweets = np.array(tweets)[order]
        labels = np.array(labels)[order]
        tweets = [torch.from_numpy(tweet) for tweet in tweets]
        tweets = pack_sequence(tweets)
        return tweets, torch.from_numpy(labels)


def infer_sleep(tweets_time, num_seconds_day, num_seconds_sleep):
    """
    infer the sleeping time of a client, by finding a length-num_seconds_sleep window with the least tweets in it
    window is a *open* interval
    return: (window_start, window_end)
    """
    tweets_time = sorted(tweets_time)

    num_tweets = len(tweets_time)
    min_included = len(tweets_time)
    start_idx_min, end_idx_min = -1, -1
    start_idx = 0       # interval start
    end_idx = (start_idx + 1) % num_tweets         # first tweet outside the open interval
    while start_idx != num_tweets:
        start_time = tweets_time[start_idx]
        end_time = tweets_time[end_idx] if end_idx > start_idx else tweets_time[end_idx] + num_seconds_day
        if end_time - start_time < num_seconds_sleep:
            end_idx = (end_idx + 1) % num_tweets
        else:   # start_idx -> end_idx - 1 are included in the interval
            if end_idx > start_idx:
                num_included = end_idx - start_idx - 1
            else:
                num_included = end_idx - start_idx - 1 + num_tweets

            if num_included < min_included:
                start_idx_min, end_idx_min = start_idx, end_idx
                min_included = num_included

            start_idx += 1

    min_start_time = tweets_time[start_idx_min]
    max_start_time = tweets_time[end_idx_min] - num_seconds_sleep
    max_start_time = max_start_time if max_start_time >= 0 else max_start_time + num_seconds_day

    if max_start_time < min_start_time:
        max_start_time += num_seconds_day
    start_time = (min_start_time + max_start_time) / 2
    end_time = start_time + num_seconds_sleep

    return start_time % num_seconds_day, end_time % num_seconds_day


class ProcessData:
    pass


class BasicProcess(ProcessData):
    """
    process
        SENTIMENT as int (1 / 0)
        DATETIME as datetime.datetime
        CLIENT_ID as unique int
    """
    def __init__(self):
        super().__init__()
        self.id = 'base'

    def __call__(self, raw_data):
        client_idmap = {}
        c_client_id = 0
        month_map = dict(Jan=1, Feb=2, Mar=3, Apr=4, May=5, Jun=6, Jul=7, Aug=8, Sep=9, Oct=10, Nov=11, Dec=12)
        for idx, (sentiment, tweet_id, dtime, query, client_id, tweet) in enumerate(raw_data):
            if sentiment == '4':
                sentiment = 1
            elif sentiment == '0':
                sentiment = 0
            else:
                raise RuntimeError("sentiment not 0 / 4")

            weekday, month, day, time_, _, year = dtime.split()
            month = month_map[month]
            day = int(day)
            hour, minute, second = [int(x) for x in time_.split(':')]
            year = int(year)
            dtime = datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second)

            if client_id not in client_idmap:
                client_idmap[client_id] = c_client_id
                c_client_id += 1
            client_id = client_idmap[client_id]

            raw_data[idx] = [sentiment, tweet_id, dtime, query, client_id, tweet]
        return raw_data
        # print(c_client_id)


class ComposedProcess(ProcessData):
    def __init__(self, *processes):
        super().__init__()
        if len(processes) == 0:
            raise ValueError("should pass at least one transform")
        self.processes = processes
        ids = [process.id for process in processes]
        self.id = '_'.join(ids)
        self.ids = ids

    def __process(self, raw_data, processes):   # recursive process for loaded data
        if len(processes) == 1:
            return processes[0](raw_data)

        ids = self.ids[:len(processes)]
        prev_id = '_'.join(ids[:-1])
        cache_fn = prev_id + '.pkl'
        cache_fp = osp.join(common.data_cache_dir, cache_fn)
        if osp.exists(cache_fp):
            raw_data = pickle.load(open(cache_fp, 'rb'))
            print(f"intermediate data loaded from {cache_fp}")
        else:
            raw_data = self.__process(raw_data, processes[:-1])
            pickle.dump(raw_data, open(cache_fp, 'wb'))
            print(f"intermediate data dumped into {cache_fp}")
        raw_data = processes[-1](raw_data)
        return raw_data

    def __call__(self, raw_data, load_cache=True):
        if load_cache:
            return self.__process(raw_data, self.processes)
        else:
            for process in self.processes:
                raw_data = process(raw_data)
            return raw_data


def get_and_fill(kwargs, key, default):
    if key not in kwargs:
        kwargs[key] = default
    return kwargs[key]


class CleanTweet(ProcessData):
    def __init__(self, verbose=False, **kwargs):
        super().__init__()
        _id = "clean"
        self.lxml = get_and_fill(kwargs, 'lxml', True)
        self.username = get_and_fill(kwargs, 'username', True)
        self.url = get_and_fill(kwargs, 'url', True)
        self.bom = get_and_fill(kwargs, 'bom', True)
        self.rm_punctuation = get_and_fill(kwargs, 'rm_punctuation', True)
        self.rm_repeat = get_and_fill(kwargs, 'rm_repeat', True)
        self.lowercase = get_and_fill(kwargs, 'lowercase', True)
        kwargs = collections.OrderedDict(kwargs)
        for key, val in kwargs.items():
            if not val:
                _id += f'_{key}'
        self.id = _id

        self.verbose = verbose

    def __call__(self, raw_data):
        user_pat = re.compile(r'@[A-Za-z0-9]+')
        http_pat = re.compile(r'https?://[A-Za-z0-9./]+')
        punctuation_pat = re.compile(r"[^a-zA-Z]")

        if self.verbose:
            print("cleaning data")
        for idx, data_tup in enumerate(raw_data):
            if self.verbose and idx % 1000 == 0:
                print(f"{idx}/{len(raw_data)}", end='\r')
            tweet = data_tup[Sentiment140Dataset.TWEET]
            if self.lxml:
                tweet = BeautifulSoup(tweet, 'lxml').get_text()
            if self.username:
                tweet = re.sub(user_pat, 'atUSERNAMEat', tweet)
            if self.url:
                tweet = re.sub(http_pat, 'URLhttp', tweet)
            if self.bom:
                tweet = tweet.replace(u"\xef\xbf\xbd", " ")
            if self.lowercase:
                tweet = tweet.lower()
            if self.rm_punctuation:
                tweet = re.sub(punctuation_pat, ' ', tweet)
            if self.rm_repeat:
                words = tweet.split()
                for word_idx, word in enumerate(words):
                    last_ch = None
                    count = 1
                    ch_list = []
                    for ch in word:
                        if count < 2 or ch != last_ch:
                            ch_list.append(ch)
                        if ch == last_ch:
                            count += 1
                        else:
                            count = 1
                        last_ch = ch
                    word = ''.join(ch_list)
                    words[word_idx] = word
                tweet = ' '.join(words)
            raw_data[idx][Sentiment140Dataset.TWEET] = tweet

        return raw_data


class BagOfWords(ProcessData):
    def __init__(self, num_features, verbose=False):
        """
        Bag of words model, take the most frequent num_features words as features.
        Tweets processed as torch.FloatTensor
        """
        super().__init__()
        self.num_features = num_features
        self.id = f"BOW{num_features}"
        self.verbose = verbose

    def __call__(self, raw_data):
        verbose = self.verbose

        # count words
        if verbose:
            print("counting words")
        words2num = {}
        for idx, data_tup in enumerate(raw_data):
            if verbose and idx % 1000 == 0:
                print(f"{idx}/{len(raw_data)}", end='\r')
            tweet = data_tup[Sentiment140Dataset.TWEET]
            words = tweet.split()
            for word in words:
                if word in words2num:
                    words2num[word] += 1
                else:
                    words2num[word] = 1
        words2num = list(words2num.items())
        words2num = sorted(words2num, key=lambda pair: -pair[1])
        valid_features = words2num[:self.num_features]
        valid_features_dict = {}
        for idx, (word, num) in enumerate(valid_features):
            valid_features_dict[word] = idx
        if verbose:
            print()

        # transform
        if verbose:
            print("transforming")
        for idx, data_tup in enumerate(raw_data):
            if verbose and idx % 1000 == 0:
                print(f"{idx}/{len(raw_data)}", end='\r')
            tweet = data_tup[Sentiment140Dataset.TWEET]
            words = tweet.split()
            idx2count = {}
            for word in words:
                if word in valid_features_dict:
                    if valid_features_dict[word] in idx2count:
                        idx2count[valid_features_dict[word]] += 1
                    else:
                        idx2count[valid_features_dict[word]] = 1
            idx2count = list(idx2count.items())
            raw_data[idx][Sentiment140Dataset.TWEET] = idx2count
        if verbose:
            print()

        return raw_data

    @staticmethod
    def multihot(num_features):
        """
        multihot decode
        """
        def _multihot(idx2count):
            vec = torch.zeros(num_features, dtype=torch.float32)
            for idx, count in idx2count:
                vec[idx] = count
            return vec
        return _multihot


def time2int(time_):
    # time_: Time or DateTime
    return time_.hour * 3600 + time_.minute * 60 + time_.second


def _test_infer_sleep():
    tweets_time = [random.randint(0, 76400 - 1) for _ in range(32)]
    print(tweets_time)
    print(infer_sleep(tweets_time, 86400, 28800))


def plot_availability_count(clients, fn="availability_count.png"):
    avai_counts = []
    for hour in range(24):
        time_ = datetime.timedelta(hours=hour)
        avai_count = 0
        for client in clients:
            if client.is_available(time_):
                avai_count += 1
        avai_counts.append(avai_count)
    plt.figure(0)
    plt.plot(range(24), avai_counts)
    plt.savefig(osp.join(common.figure_fd, fn))
    plt.close('all')


def _plot_block_count2client_count(clients):
    block_count2client_count = dict()
    for client in clients:
        block_count = len(client.available_blocks)
        if block_count not in block_count2client_count:
            block_count2client_count[block_count] = 0
        block_count2client_count[block_count] += 1
    plot_array = sorted(list(block_count2client_count.items()), key=lambda x: x[0])
    _x, _y = list(zip(*plot_array))
    plt.figure(1)
    plt.plot(_x, _y)
    plt.savefig(osp.join(common.figure_fd, "block_count2client_count.png"))
    plt.close('all')


def plot_client_sample_volume(clients):
    client_num_samples = [client.num_samples for client in clients]
    max_num_samples = max(client_num_samples)
    x_axis = np.arange(max_num_samples + 1)
    y_axis = np.zeros_like(x_axis)
    for num_samples in client_num_samples:
        y_axis[num_samples] += 1
    summ = 0
    y_axis_accumulate = np.zeros_like(x_axis)
    for num_samples, num_clients in enumerate(y_axis):
        summ += num_clients
        print(f"num_samples: {num_samples}, num_clients: {num_clients}, total: {summ}")
        y_axis_accumulate[num_samples] = summ
    fig = plt.figure(0)
    plt.plot(x_axis[1:], y_axis[1:])
    plt.savefig(osp.join(common.figure_fd, "num_samples_distribution.png"))
    plt.close(0)
    fig = plt.figure(1)
    plt.plot(x_axis[1:], y_axis_accumulate[1:])
    plt.savefig(osp.join(common.figure_fd, "num_samples_pdf.png"))
    plt.close(1)


def _plot_availability_pattern(dataset, clients):
    num_figures = 9
    test_num = 1
    thres_num_samples = 10
    filtered_clients = [client for client in clients if client.num_samples > thres_num_samples]
    datetimes = [tup[Sentiment140Dataset.DATETIME] for tup in dataset.processed_data]
    start_date = min(datetimes).date()
    plt.figure(figsize=(14, 14))
    for figure_idx in range(num_figures):
        plt.subplot(3, 3, figure_idx + 1)
        client_indices = random.sample(range(len(filtered_clients)), test_num)
        print(f"chosen clients: {client_indices}")
        test_clients = [filtered_clients[idx] for idx in client_indices]
        date_line = []
        time_line = []
        for client in test_clients:
            date_line_ = []
            time_line_ = []
            for idx in sorted(client.train_indices):
                data_tup = dataset.processed_data[idx]
                dtime = data_tup[Sentiment140Dataset.DATETIME]
                this_date = dtime.date()
                this_time = dtime.time()
                date_line_.append((this_date - start_date).days)
                time_line_.append(this_time.hour * 3600 + this_time.minute * 60 + this_time.second)
            date_line_ = np.array(date_line_)
            time_line_ = np.array(time_line_)
            order = date_line_.argsort()
            date_line.append(date_line_[order])
            time_line.append(time_line_[order])

        for dline, tline in zip(date_line, time_line):
            plt.scatter(dline, tline)
        y_ticks_texts = list(range(25))
        y_ticks_vals = [text * 3600 for text in y_ticks_texts]
        plt.yticks(y_ticks_vals, y_ticks_texts)
    plt.savefig(osp.join(common.figure_fd, f"availability.png"))
    plt.close('all')


def _statistics(dataset):
    dataset.statistic_words(num_print=0)
    dataset.count_in_gloves()
    print(f"Number of clients: {dataset.count_clients()}")
    print(f"Number of words: {dataset.count_words()}")
    print(f"Number of characters: {dataset.count_chars()}")
    clients = dataset.get_clients(train_batch_size=1)
    client_num_samples = [client.num_samples for client in clients]
    max_num_samples = max(client_num_samples)
    print(f"maximum number of samples {max_num_samples}")
    positive_count = 0
    for tweet, label in dataset:
        if label == 1:
            positive_count += 1
    print(f"{positive_count}/{len(dataset)} positive samples")


def _test_forward(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True,
                                             collate_fn=Sentiment140Dataset.collate_fn)
    net = LSTM2()
    for tweets, labels in dataloader:
        print(tweets)
        print(labels)
        output = net(tweets)
        output.sum().backward()
        print(output.shape)
        break


def _test_filter(dataset):
    dataset.filter_clients_(8)
    dataset.random_select_clients_(10000)
    trainset, testset = dataset.partition()
    actual_N = trainset.count_clients()
    print(f"There are actually {actual_N} clients")


def _count_availability_mid16(dataset):
    thres = 100
    dataset.filter_clients_(thres)

    clients = dataset.get_clients(strategy="modeled_mid")
    print(f"{len(clients)} clients")
    avai_counts = []
    for hour in np.arange(0, 24, 0.5):
        time_ = datetime.timedelta(seconds=hour*3600)
        avai_count = 0
        for client in clients:
            if client.is_available(time_):
                avai_count += 1
        avai_counts.append(avai_count)
    plt.figure(0)
    plt.plot(np.arange(0, 24, 0.5), avai_counts)
    plt.savefig(osp.join(common.figure_fd, f"available_count_modeled_mid_filter{thres}.png"))
    plt.close('all')


def plot_datum_time(dataset, fn="tweets_time.png"):
    counts = [0] * 24
    for data_tup in dataset.processed_data:
        datum_time = data_tup[dataset.DATETIME]
        datum = datum_time.hour
        counts[datum] += 1
    plt.figure(0)
    plt.plot(range(24), counts)
    plt.savefig(osp.join(common.figure_fd, fn))
    plt.close('all')


def plot_posi_ratio_with_time(dataset):
    posi_counts = np.zeros(24, int)
    total_counts = np.zeros(24, int)
    for data_tup in dataset.processed_data:
        hour = data_tup[dataset.DATETIME].hour
        sentiment = data_tup[dataset.SENTIMENT]
        total_counts[hour] += 1
        if sentiment == 1:
            posi_counts[hour] += 1
    plt.figure(0)
    plt.plot(range(24), total_counts)
    plt.plot(range(24), posi_counts)
    plt.savefig(osp.join(common.figure_fd, "count_in_variation.png"))
    plt.close('all')
    plt.figure(1)
    plt.plot(range(24), posi_counts / total_counts)
    plt.savefig(osp.join(common.figure_fd, "ratio_in_variation.png"))
    plt.close('all')
    print(f"{sum(posi_counts)}/{sum(total_counts)} tweets are positive")


def _test():
    dataset = Sentiment140Dataset(ComposedProcess(
        BasicProcess(),
        # CleanTweet(verbose=True),
    ))
    dataset.filter_clients_(40)
    print(dataset.count_clients())
    print(len(dataset))
    exit()
    dataset.random_select_clients_(1000)
    dataset.time_variation_(num_blocks=24)
    trainset, testset = dataset.partition()
    print(trainset.count_clients())
    _plot_posi_ratio_with_time(trainset)


if __name__ == "__main__":
    _test()
