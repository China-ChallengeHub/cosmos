""" Define the sublayers in encoder/decoder layer """
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .Modules import ScaledDotProductAttention

__author__ = "Yu-Hsiang Huang"


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        # n_head:  4
        # d_model: 768
        # d_k:     192
        # d_v:     192
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        # self.layer_norm = nn.LayerNorm(d_model)
        self.layer_norm = BertLayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # mask = slf_attn_mask
        # q: bsz * len_q * d_model
        # k: bsz * len_k * d_model
        # v: bsz * len_v * d_model

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        # q: bsz * len_q * n_head * d_k
        # k: bsz * len_k * n_head * d_k
        # v: bsz * len_v * n_head * d_v
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # q: (n_head * bsz) * len_q * d_k
        # k: (n_head * bsz) * len_k * d_k
        # v: (n_head * bsz) * len_v * d_v
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        # 将所有的mask矩阵统一输入为如下形式
        # mask: (n_head * bsz) * len_q * len_q

        # output: (n_head * bsz) * len_q * d_v
        # attn:   (n_head * bsz) * len_q * len_k
        output, attn = self.attention(q, k, v, mask=mask)

        # output: n_head * bsz * len_q * d_v
        # output: bsz * n_head * len_q * d_v
        # output: n_head * bsz * len_q * d_v

        # output: bsz * len_q * n_head * d_v
        # output: bsz * len_q * (n_head * d_v)
        # output: bsz * len_q * d_model
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        # 对每一个单词的embedding向量进行归一化操作
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    """
    Notes: Normalization作用为将激活函数后的值规整为均值为0,方差为1的正态分布
    """
    def __init__(self, d_in, d_hid, dropout=0.1):
        # d_in: 768
        # d_model: 3072
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)  # position-wise

        # self.layer_norm = nn.LayerNorm(d_in)
        self.layer_norm = BertLayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        # 对每一个单词的embedding向量进行归一化操作
        output = self.layer_norm(output + residual)
        return output
