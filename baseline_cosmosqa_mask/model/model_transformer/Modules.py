import torch
import torch.nn as nn
import numpy as np

__author__ = "Yu-Hsiang Huang"


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)  # dim=-1

    def forward(self, q, k, v, mask=None):
        # q: (n_head * bsz) * len_q * d_k
        # k: (n_head * bsz) * len_k * d_k
        # v: (n_head * bsz) * len_v * d_v
        # attn: (n_head * bsz) * len_q * len_k
        # attn_score = (qk/(k^0.5))
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            # attn = attn.masked_fill(mask, -np.inf)

            mask = mask.float()
            attn = attn.mul(mask)

            # debug: 绝不可添加mask.byte()
            # debug: 填充为0还是-1e9,若填充为-1e9,则效果会变差
            # attn = attn.masked_fill(mask == 0, -1e9)

        # attn: (n_head * bsz) * len_q * len_k
        # output: (n_head * bsz) * len_q * d_v
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn
