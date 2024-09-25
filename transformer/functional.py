# -*- coding: utf-8 -*-
# date: 2018-11-29 20:07
import copy
import math
from copy import deepcopy

import numpy
import torch
import torch.nn as nn
from torch.nn.functional import softmax
import numpy as np
import os

def clones(module, n):
    """
    Produce N identical layers.
    """
    assert isinstance(module, nn.Module)
    return nn.ModuleList([deepcopy(module) for _ in range(n)])


def subsequent_mask(size):
    """
    Mask out subsequent positions.
    """
    attn_shape = (1, size, size)
    mask = numpy.triu(numpy.ones(attn_shape), k=1).astype('uint8')

    # print(attn_shape)
    # print("="*10)
    # print(mask)
    # os.system("PAUSE")
    return torch.from_numpy(mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scaled Dot Product Attention'
        q k v =[batchSize, 8, 7, 64]; mask =[batchSize, 1， 1, 7]
        q k v =[batchSize, 8, 12, 64]; mask =[batchSize,1, 12, 12]

        cross attn:
        q = [batchSize, 8, 12, 64]
        k v =[batchSize,8, 7, 64]
        mask = [batSize,1，1, 7]
    """
    # print(numpy.shape(query))
    # print(numpy.shape(key.transpose(-2, -1)))

    d_k = query.size(-1) #[batSize,8, 7, 64]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  #[batSize, h, 7,7] or [batSize, h, 12,12]； cross: [batSize, 8, 12, 7]
    # print(numpy.shape(scores))
    # print(scores[0][0])
    # print("="*10)
    # print(numpy.shape(mask))
    # print((mask[0][0]))
    # temp = scores.clone()
    if mask is not None:
        scores = scores.masked_fill_(mask == 0, value=-1e9)
    # print(scores[0][0])
    # # # print(temp is scores)
    # print(temp.equal(scores))
    # os.system("PAUSE")
    p_attn = softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn     #[batSize, h, 7,64]
