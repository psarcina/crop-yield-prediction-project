#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 12:08:51 2020

@author: pasq
"""

import torch
from torch import nn

class Chomp3d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp3d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size, :, :].contiguous()



class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, filters, kernel_size, stride, padding, dilation):
        super(TemporalBlock, self).__init__()
        f1, f2, f3 = filters
        k1, k2, k3 = kernel_size
        p1, p2, p3 = padding

        self.conv1 = nn.Conv3d(n_inputs, f1, (1, k2, k3),
                               stride=stride, padding=(0,p2,p3))
        self.bn1 = nn.BatchNorm3d(f1)
        self.relu1 = nn.ReLU()
#        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv3d(f1, f2, (1, k2, k3),
                               stride=stride, padding=(0,p2,p3))
        self.bn2 = nn.BatchNorm3d(f2)
        self.relu2 = nn.ReLU()
#        self.dropout2 = nn.Dropout(dropout)
        
        self.conv3 = nn.Conv3d(f2, f3, (k1, k2, k3),
                               stride=stride, padding=(p1,p2,p3), dilation=(dilation,1,1))
        self.chomp = Chomp3d(p1)
        self.bn3 = nn.BatchNorm3d(f3)
        self.relu3 = nn.ReLU()

        self.net = nn.Sequential(self.conv1, self.bn1, self.relu1,
                                 self.conv2, self.bn2, self.relu2,
                                 self.conv3, self.chomp, self.bn3, self.relu3)
#        self.downsample = nn.Conv3d(n_inputs, f3, 1) if n_inputs != f3 else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
#        if self.downsample is not None:
#            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
#        res = x if self.downsample is None else self.downsample(x)
        return out


class TemporalOutput(nn.Module):
    def __init__(self, filters):
        super().__init__()
        f1, f2 = filters
        self.flat = nn.Flatten()
        self.lin1 = nn.Linear(f1, f2)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(f2, 1)
        
    def forward(self, x):
        x = x[:,:,-1,:,:]
        x = self.flat(x)
        x = self.relu(self.lin1(x))
        x = self.lin2(x)
        return x



TemporalConvNet = nn.Sequential(
        TemporalBlock(4, (10, 25, 50), kernel_size=(3,7,7), stride=(1,2,2), padding=(2,3,3), dilation=1),
        TemporalBlock(50, (128, 256, 512), kernel_size=(3,7,7), stride=(1,2,2), padding=(4,3,3), dilation=2),
        TemporalBlock(512, (1024,2048,2048), kernel_size=(3,5,5), stride=(1,2,2), padding=(8,3,3), dilation=4),
        TemporalOutput((18432, 2048))
        )
