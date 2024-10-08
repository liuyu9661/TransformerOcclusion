# -*- coding: utf-8 -*-
# date: 2018-11-30 15:27
import torch.nn as nn

from .functional import clones
from .sublayer_connection import SublayerConnection
import numpy as np
import os

class EncoderLayer(nn.Module):
    """
    Encoder is made up of self-attn and feed forward (defined below)
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """
        Follow Figure 1 (left) for connections.
        X:    [batSize,7,2] => [batSize,7,512]
        mask: [batSize,1,7]
        """
        # print(np.shape(x))
        # print(np.shape(mask))
        # print(mask[0])
        # os.system("PAUSE")
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
