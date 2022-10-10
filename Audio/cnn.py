#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 06:34:39 2022

@author: vlourenco
"""

from torch import nn, device
from torchsummary import summary

class CNNNetwork(nn.Module):
    
    def __init__(self):
        super().__init__()
        # 4 conv blocks / flatten / linear / softmax
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, #number of input channels
                out_channels=16, #number of filters , output channels
                kernel_size=3,
                stride=1,
                padding=2
                ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16, #number of input channels
                out_channels=32, #number of filters , output channels
                kernel_size=3,
                stride=1,
                padding=2
                ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, #number of input channels
                out_channels=64, #number of filters , output channels
                kernel_size=3,
                stride=1,
                padding=2
                ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, #number of input channels
                out_channels=128, #number of filters , output channels
                kernel_size=3,
                stride=1,
                padding=2
                ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            )
        
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(14720, 4) #Shape of the data output from the last layer
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        predictions = self.softmax(logits)
        
        return predictions

if __name__ == "__main__":
    #device = torch.device("mps")
    cnn = CNNNetwork()
    #cnn.to(device=device)
    summary(cnn, (1, 64, 352)) #(channels, frequency axis, time axis)#
    
    
    
        
