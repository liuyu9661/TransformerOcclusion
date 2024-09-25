# -*- coding: utf-8 -*-
# date: 2018-11-30 16:35
import torch.nn as nn
import numpy as np
import os

from .functional import clones, attention


class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        """
        Take in model size and number of heads.
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        #  We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        Implements Figure 2
        q k v =[batchSize, 7, 512]
        q k v =[batchSize, 12, 512]

        cross attn:
        q = [batchSize, 12, 512]
        k v =[batchSize, 7, 512]
        mask = [batSize,1,7]
        """
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                             zip(self.linears, (query, key, value))]
        # q k v [batSize,8, 7, 64]  or q k v [batSize,8, 12, 64]
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout) #x[batsize,8,7,64] attn[batsize,8,7,7] or x[batsize,8,12,64] attn[batsize,8,12,12] or cross x[batsize,8,12,64]
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k) #x[batsize,7,512]
        return self.linears[-1](x)
