#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 18:15:09 2022

@author: vlourenco
"""

import EEGNet
from EEGNet import EEGNet_torch_test
from operations import prepare_data
from Organized.tools import use_wandb
import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm
import wandb
import os

### PARAMETERS ###
USE_WB = 0
NUM_EPOCHS = 1
MODEL_NAME = "eegnet_epoch_1"

def eval(test_loader, model, device):

    model.eval()
    num_correct = 0
    num_sample = 0
    with torch.no_grad():
        for batch_test in test_dataloader:
            # get the inputs; data is a list of [inputs, labels]
            inputs = batch_test[0].to(device)
            labels = batch_test[1].to(device)
                
            #input_ = torch.zeros([2,129,2500,1])
            input_ = torch.zeros([2,129,2500,1])
            input_ = torch.unsqueeze(inputs,dim=3)
        
            # forward + backward + optimize
            outputs = model(input_)
            _, predicted = torch.max(outputs.data, 1)
            num_sample += labels.size(0)
            num_correct += (predicted == labels).sum().item()
        testing_acc = 100 * num_correct / num_sample
        #print(f'Accuracy:{testing_acc}')
        
    return testing_acc

if __name__ == "__main__":
    device = "cpu"
    if USE_WB == 1:
        use_wandb(projectname="test")

    #Prepare Data
    test_dataloader, testloader = prepare_data(file="/Users/vlourenco/Documents/GitHub/EEG_MIB/Organized/P1_a1.mat")
    
    model = EEGNet_torch_test()
    model.load_state_dict(torch.load(f'/Users/vlourenco/Documents/GitHub/EEG_MIB/Organized/Saved_models/{MODEL_NAME}.pt'))
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    
          
    print('Finished Testing')