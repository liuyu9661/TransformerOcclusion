# -*- coding: utf-8 -*-
# date: 2018-11-30 15:41
import torch.nn as nn

from .functional import clones
from .sublayer_connection import SublayerConnection
import os
import numpy as np

class DecoderLayer(nn.Module):
    """
    Decoder is made of self-attn, src-attn, and feed forward (defined below)
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        Follow Figure 1 (right) for connections.
        memory: [batchSize, 7, 512]
        src_mask: [batSize,1,7]
        x: [batchSize,12,512]
        tgt_mask: [batchSize,12,12]
        """
        m = memory #[batchSize, 7, 512],
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))   #[batchSize, 12, 512],
        # print("==Cross Attn==")
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))    #[batchSize, 12, 512],
        # print(np.shape(x))
        # os.system("PAUSE")
        return self.sublayer[2](x, self.feed_forward)
