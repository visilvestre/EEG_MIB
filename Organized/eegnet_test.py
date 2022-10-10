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
                                                  out_channels = 8,               #In tensorflow => Filters
                                                  kernel_size = (1, kernLength),
                                                  stride=1,
                                                  padding = 'same',
                                                  dilation= 1,
                                                  groups = 2,
                                                  bias = False,
                                                  padding_mode = 'zeros',
                                                  device = None,
                                                  dtype = None),
                                        nn.BatchNorm2d(8,
                                                       eps=1e-03,
                                                       momentum=0.99),
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
                                        nn.BatchNorm2d(8,
                                                       eps=1e-03,
                                                       momentum=0.99), 
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
                 depth_multiplier = 2,
                 nb_classes = 2
                ):
        super(EEGNet_torch_test, self).__init__()
        intermediate_channels = 8 * depth_multiplier
        self.sequential = nn.Sequential(nn.Conv2d(in_channels = 129, 
                                                  out_channels = 8, 
                                                  kernel_size = (1, kernLength), 
                                                  padding = 'same',
                                                  dilation= 1,
                                                  groups = 1,
                                                  bias = False,
                                                  padding_mode = 'zeros',
                                                  device = None,
                                                  dtype = None),
                                        nn.BatchNorm2d(8), 
                                        nn.Conv2d(in_channels = 8, 
                                                  out_channels = intermediate_channels, 
                                                  kernel_size = (129, 1), 
                                                  stride = (1,1), 
                                                  padding = 'same',
                                                  dilation = 1,
                                                  groups= 8,
                                                  bias = False, 
                                                  padding_mode = 'zeros', 
                                                  device = None, 
                                                  dtype = None),
                                        nn.BatchNorm2d(16),
                                        nn.ELU(alpha=1.0, 
                                               inplace=False), 
                                        nn.AvgPool2d((1,1),
                                                     stride=None, 
                                                     padding=0, 
                                                     ceil_mode=False, 
                                                     count_include_pad=True, 
                                                     divisor_override=None),
                                        nn.Dropout(dropoutRate), 
                                        torch.nn.Conv2d(in_channels= 16,
                                                        out_channels= intermediate_channels,
                                                        kernel_size=(16,1),
                                                        stride=1,
                                                        padding=0,
                                                        dilation=1,
                                                        groups=8,
                                                        bias=False,
                                                        padding_mode='zeros'), 
                                        torch.nn.Conv2d(in_channels= intermediate_channels,
                                                        out_channels= 16,
                                                        kernel_size=(16,1),
                                                        stride=1,
                                                        padding=0,
                                                        dilation=1,
                                                        bias=False,
                                                        padding_mode='zeros'), 
                                        nn.BatchNorm2d(num_features = 16), 
                                        nn.ELU(alpha=1.0, inplace=False),
                                        nn.AvgPool2d((1,1), 
                                                     stride=None, 
                                                     padding=0, 
                                                     ceil_mode=False, 
                                                     count_include_pad=True, 
                                                     divisor_override=None), 
                                        nn.Dropout(dropoutRate), 
                                        nn.Flatten(), 
                                        nn.Linear(in_features = 39520,
                                                  out_features = 2, 
                                                  bias=True, 
                                                  device=None, 
                                                  dtype=None),
                                        nn.Softmax(dim=None) )
        
    
    def forward(self, x):
        softmax = self.sequential(x)
        return softmax
    
class EEGNet(nn.Module):
    
    #from github: https://github.com/aliasvishnu/EEGNet/blob/master/EEGNet-PyTorch.ipynb
    
    def __init__(self):
        super(EEGNet, self).__init__()
        self.T = 120
        
        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, 64), padding = 0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)
        
        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)
        
        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))
        
        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 120 timepoints. 
        self.fc1 = nn.Linear(4*2*7, 1)
        

    def forward(self, x):
        # Layer 1
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        x = x.permute(0, 3, 1, 2)
        
        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)
        
        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)
        
        # FC Layer
        x = x.view(-1, 4*2*7)
        x = F.sigmoid(self.fc1(x))
        return x