import copy

import torch.nn as nn
import torch
from torch.nn.parameter import Parameter

import common
import os.path as osp
import pickle


def load_embedding(fn, model_fd):
    def _init(m):
        fp = osp.join(model_fd, fn)
        weight = pickle.load(open(fp, 'rb'))
        m.weight.data[...] = torch.from_numpy(weight)
    return _init


class Dice(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.weight = Parameter(torch.Tensor(num_features).fill_(0.25), requires_grad=True)
        self.bn = nn.BatchNorm1d(num_features=num_features, momentum=0.0, eps=1e-9, affine=False)

    def forward(self, inp):
        inp_normed = self.bn(inp)
        prob = torch.sigmoid(inp_normed)
        inp = prob * inp + (1 - prob) * inp * self.weight
        return inp


class MLP(nn.Module):
    def __init__(self, embedding_dim, hidden_size=(200, 80), activation='PReLU'):
        super().__init__()

        input_size = embedding_dim * 9
        # self.bn = nn.BatchNorm1d(num_features=input_size, momentum=0.99, eps=0.001)
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], 2)

        if activation == 'PReLU':
            self.a1 = nn.PReLU(hidden_size[0])
            self.a2 = nn.PReLU(hidden_size[1])
        elif activation == 'Dice':
            self.a1 = Dice(hidden_size[0])
            self.a2 = Dice(hidden_size[1])
        else:
            raise ValueError(f"Unrecognized activation: {activation}")

    def forward(self, x):
        # x = self.bn(x)
        x = self.a1(self.fc1(x))
        x = self.a2(self.fc2(x))
        x = self.fc3(x)
        return x


class UserEmbeddingAttentionLayer(nn.Module):
    def __init__(self, embedding_dim, hidden_size=(80, 40), activation=torch.sigmoid):
        super().__init__()

        self.fc1 = nn.Linear(embedding_dim * 8, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], 1)
        self.activation = activation

    def forward(self, queries, user_behaviors, masks):
        """
        :param queries: (B, 2D)
        :param user_behaviors: (B, L, 2D) L is the maximum length
        :param masks: (B, L) 1 for valid behavior (0 for padded)
        :return: user embeddings (B, 2D)
        """
        # (B, L, 2D)
        queries = queries.unsqueeze(1).repeat(1, user_behaviors.shape[1], 1)
        B, L, _D = queries.shape
        D = _D // 2

        # initial scores for attention input
        scores = torch.cat([queries, user_behaviors, queries-user_behaviors, queries*user_behaviors], dim=2)
        scores = scores.view(-1, D * 8)
        scores = self.activation(self.fc1(scores))
        scores = self.activation(self.fc2(scores))
        # B L 1
        scores = self.fc3(scores)
        scores = scores.view(B, L, 1)

        # softmax the score
        masks = masks.unsqueeze(-1).to(dtype=torch.bool)
        paddings = torch.ones_like(scores) * (-2 ** 32 + 1)
        # zero out padded behaviors
        scores = torch.where(masks, scores, paddings)
        scores = torch.nn.functional.softmax(scores, dim=1)

        # apply attention and mask
        user_behaviors = scores * user_behaviors
        user_behaviors = user_behaviors.sum(dim=1)

        return user_behaviors


class DeepInterestNetwork(nn.Module):
    def __init__(self, num_users=49023, num_goods=143534, num_cats=4815, embedding_dim=18, activation='PReLU'):
        super().__init__()
        self.num_users, self.num_goods, self.num_cats, self.embedding_dim = \
            num_users, num_goods, num_cats, embedding_dim
        self.activation = activation

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.good_embedding = nn.Embedding(num_goods, embedding_dim)
        self.cat_embedding = nn.Embedding(num_cats, embedding_dim)
        self.mlp = MLP(embedding_dim=embedding_dim, activation=activation)
        self.attention = UserEmbeddingAttentionLayer(embedding_dim=embedding_dim)

    def forward(self, uid_b, gid_b, cid_b, gid_his_b, cid_his_b, masks):
        """
        :param uid_b: (B) user ids batch
        :param gid_b: (B) good ids batch
        :param cid_b: (B) category ids batch
        :param gid_his_b: (B, L) interacted good ids batch, padded
        :param cid_his_b: (B, L) interacted category ids batch, padded
        :param masks: (B, L) mask for interacted goods
        :return: CTR (direct output of self.mlp, not softmaxed)
        """
        user_b = self.user_embedding(uid_b)
        good_b = self.good_embedding(gid_b)
        cat_b = self.cat_embedding(cid_b)
        good_his_b = self.good_embedding(gid_his_b)
        cat_his_b = self.cat_embedding(cid_his_b)

        query_b = torch.cat([good_b, cat_b], dim=-1)
        user_behaviors = torch.cat([good_his_b, cat_his_b], dim=-1)
        attention_output = self.attention(query_b, user_behaviors, masks)

        user_behaviors_sum_b = user_behaviors.sum(dim=1)
        mlp_input = torch.cat([user_b, query_b, user_behaviors_sum_b, user_behaviors_sum_b*query_b, attention_output],
                              dim=-1)
        mlp_output = self.mlp(mlp_input)
        mlp_output = mlp_output    # to compat the loaded weight from tensorflow
        return mlp_output


class DeepInterestNetworkFixEmb(nn.Module):
    def __init__(self, din_source: DeepInterestNetwork, copy_pars=False):
        super().__init__()
        self.embeddings = dict(user=None, good=None, cat=None)
        if copy_pars:
            self.embeddings['user'] = copy.deepcopy(din_source.user_embedding)
            self.embeddings['good'] = copy.deepcopy(din_source.good_embedding)
            self.embeddings['cat'] = copy.deepcopy(din_source.cat_embedding)
            self.mlp = copy.deepcopy(din_source.mlp)
            self.attention = copy.deepcopy(din_source.attention)
        else:
            ub_fn, gb_fn, cb_fn = "user_embedding.pkl", "good_embedding.pkl", "cat_embedding.pkl"
            model_fd = common.cache_fd
            self.embeddings['user'] = nn.Embedding(din_source.num_users, din_source.embedding_dim)
            self.embeddings['good'] = nn.Embedding(din_source.num_goods, din_source.embedding_dim)
            self.embeddings['cat'] = nn.Embedding(din_source.num_cats, din_source.embedding_dim)
            self.embeddings['user'].apply(load_embedding(ub_fn, model_fd=model_fd))
            self.embeddings['good'].apply(load_embedding(gb_fn, model_fd=model_fd))
            self.embeddings['cat'].apply(load_embedding(cb_fn, model_fd=model_fd))
            for _, emb in self.embeddings.items():
                emb.to(device=common.device)
                for par in emb.parameters():
                    par.requires_grad = False
            self.mlp = MLP(embedding_dim=din_source.embedding_dim, activation=din_source.activation)
            self.attention = UserEmbeddingAttentionLayer(embedding_dim=din_source.embedding_dim)

    def forward(self, uid_b, gid_b, cid_b, gid_his_b, cid_his_b, masks):
        """
        :param uid_b: (B) user ids batch
        :param gid_b: (B) good ids batch
        :param cid_b: (B) category ids batch
        :param gid_his_b: (B, L) interacted good ids batch, padded
        :param cid_his_b: (B, L) interacted category ids batch, padded
        :param masks: (B, L) mask for interacted goods
        :return: CTR (direct output of self.mlp, not softmaxed)
        """
        user_b = self.embeddings["user"](uid_b)
        good_b = self.embeddings["good"](gid_b)
        cat_b = self.embeddings['cat'](cid_b)
        good_his_b = self.embeddings["good"](gid_his_b)
        cat_his_b = self.embeddings["cat"](cid_his_b)

        query_b = torch.cat([good_b, cat_b], dim=-1)
        user_behaviors = torch.cat([good_his_b, cat_his_b], dim=-1)
        attention_output = self.attention(query_b, user_behaviors, masks)

        user_behaviors_sum_b = user_behaviors.sum(dim=1)
        mlp_input = torch.cat([user_b, query_b, user_behaviors_sum_b, user_behaviors_sum_b * query_b, attention_output],
                              dim=-1)
        mlp_output = self.mlp(mlp_input)
        mlp_output = mlp_output  # to compat the loaded weight from tensorflow
        return mlp_output


def _test_attention():
    B = 3
    D = 18
    L = 5
    queries = torch.rand((B, 2*D))
    user_behaviors = torch.rand((B, L, 2*D))
    masks = torch.ones((B, L))
    masks[:, 3:] = 0
    user_embedding = UserEmbeddingAttentionLayer(embedding_dim=D)
    user_embedding(queries, user_behaviors, masks)


def _test_din():
    din = DeepInterestNetwork(num_users=10, num_goods=11, num_cats=12)
    uids = torch.LongTensor([0, 1, 2])
    gids = torch.LongTensor([1, 2, 3])
    cids = torch.LongTensor([0, 0, 0])
    gid_his = torch.LongTensor([[3, 4, 5], [1, 2, 0], [9, 0, 0]])
    cid_his = torch.LongTensor([[1, 2, 3], [4, 5, 0], [5, 0, 0]])
    masks = torch.LongTensor([[1, 1, 1], [1, 1, 0], [1, 0, 0]])
    print(din(uids, gids, cids, gid_his, cid_his, masks))


def _test_dice():
    import numpy as np
    dice = Dice(4)
    inp = torch.from_numpy(np.arange(12).reshape(3, 4).astype(np.float32))
    out = dice(inp)


if __name__ == "__main__":
    _test_dice()