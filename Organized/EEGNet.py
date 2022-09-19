#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 14:38:55 2022

@author: vlourenco
"""


import torch
from torch import nn as nn


class EEGNet_torch(nn.Module):
    #EEGNet() Model in pytorch
    
    def __init__(self,
                 in_channels = 2,
                 F1 = 8,
                 kernLength = 500,
                 device_type = "cpu",
                 dropoutRate = 0.5,
                 depth_multiplier = 1,
                 nb_classes = 2
                ):
        super(EEGNet_torch, self).__init__()
        intermediate_channels = 8 * depth_multiplier
        self.sequential = nn.Sequential(nn.Conv2d(in_channels = 2, 
                                                  out_channels = 8, 
                                                  kernel_size = (1, kernLength), 
                                                  padding = 'same',
                                                  dilation= 1,
                                                  groups = 2,
                                                  bias = False,
                                                  padding_mode = 'zeros',
                                                  device = None,
                                                  dtype = None), 
                                        nn.BatchNorm2d(8), 
                                        nn.Conv2d(in_channels = 8, 
                                                  out_channels = 8, 
                                                  kernel_size = (1, kernLength), 
                                                  stride = (1,1), 
                                                  padding = 'same',
                                                  dilation = 1,
                                                  groups= 8,
                                                  bias = False, 
                                                  padding_mode = 'zeros', 
                                                  device = None, 
                                                  dtype = None),
                                        nn.BatchNorm2d(8), 
                                        nn.ELU(alpha=1.0, 
                                               inplace=False), 
                                        nn.AvgPool2d((1,4), 
                                                     stride=None, 
                                                     padding=0, 
                                                     ceil_mode=False, 
                                                     count_include_pad=True, 
                                                     divisor_override=None),
                                        nn.Dropout(dropoutRate), 
                                        torch.nn.Conv2d(in_channels= 8,
                                                        out_channels= intermediate_channels,
                                                        kernel_size=3,
                                                        stride=1,
                                                        padding=0,
                                                        dilation=1,
                                                        groups=8,
                                                        bias=False,
                                                        padding_mode='zeros'), 
                                        torch.nn.Conv2d(in_channels= intermediate_channels,
                                                        out_channels= 8,
                                                        kernel_size=1,
                                                        stride=1,
                                                        padding=0,
                                                        dilation=1,
                                                        bias=False,
                                                        padding_mode='zeros'), 
                                        nn.BatchNorm2d(num_features = 8), 
                                        nn.ELU(alpha=1.0, inplace=False), 
                                        nn.AvgPool2d((1,8), 
                                                     stride=None, 
                                                     padding=0, 
                                                     ceil_mode=False, 
                                                     count_include_pad=True, 
                                                     divisor_override=None), 
                                        nn.Dropout(dropoutRate), 
                                        nn.Flatten(), 
                                        nn.Linear(in_features = 78232,
                                                  out_features = 2, 
                                                  bias=True, 
                                                  device=None, 
                                                  dtype=None),
                                        nn.Softmax(dim=None) )
        
    
    def forward(self, x):
        softmax = self.sequential(x)
        return softmax
    

class EEGNet_torch_test(torch.nn.Module):
    #EEGNet() Model in pytorch
    
    def __init__(self,
                 in_channels = 2,
                 F1 = 8,
                 kernLength = 500,
                 device_type = "cpu",
                 dropoutRate = 0.5,
                 depth_multiplier = 1,
                 nb_classes = 2
                ):
        super(self, EEGNet_torch).__init__()
        intermediate_channels = 8 * depth_multiplier
        self.sequential = nn.Sequential(nn.Conv2d(in_channels = 2, 
                                                  out_channels = 8, 
                                                  kernel_size = (1, kernLength), 
                                                  padding = 'same',
                                                  dilation= 1,
                                                  groups = 2,
                                                  bias = False,
                                                  padding_mode = 'zeros',
                                                  device = None,
                                                  dtype = None), 
                                        nn.BatchNorm2d(8), 
                                        nn.Conv2d(in_channels = 8, 
                                                  out_channels = 8, 
                                                  kernel_size = (1, kernLength), 
                                                  stride = (1,1), 
                                                  padding = 'same',
                                                  dilation = 1,
                                                  groups= 8,
                                                  bias = False, 
                                                  padding_mode = 'zeros', 
                                                  device = None, 
                                                  dtype = None),
                                        nn.BatchNorm2d(8), 
                                        nn.ELU(alpha=1.0, 
                                               inplace=False), 
                                        nn.AvgPool2d((1,4), 
                                                     stride=None, 
                                                     padding=0, 
                                                     ceil_mode=False, 
                                                     count_include_pad=True, 
                                                     divisor_override=None),
                                        nn.Dropout(dropoutRate), 
                                        torch.nn.Conv2d(in_channels= 8,
                                                        out_channels= intermediate_channels,
                                                        kernel_size=3,
                                                        stride=1,
                                                        padding=0,
                                                        dilation=1,
                                                        groups=8,
                                                        bias=False,
                                                        padding_mode='zeros'), 
                                        torch.nn.Conv2d(in_channels= intermediate_channels,
                                                        out_channels= 8,
                                                        kernel_size=1,
                                                        stride=1,
                                                        padding=0,
                                                        dilation=1,
                                                        bias=False,
                                                        padding_mode='zeros'), 
                                        nn.BatchNorm2d(num_features = 8), 
                                        nn.ELU(alpha=1.0, inplace=False), 
                                        nn.AvgPool2d((1,8), 
                                                     stride=None, 
                                                     padding=0, 
                                                     ceil_mode=False, 
                                                     count_include_pad=True, 
                                                     divisor_override=None), 
                                        nn.Dropout(dropoutRate), 
                                        nn.Flatten(), 
                                        nn.Linear(in_features = 78232,
                                                  out_features = 2, 
                                                  bias=True, 
                                                  device=None, 
                                                  dtype=None),
                                        nn.Softmax(dim=None) )
        
    
    def forward(self, x):
        softmax = self.sequential(x)
        return softmax