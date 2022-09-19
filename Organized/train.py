#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 16:52:28 2022

@author: vlourenco
"""

from EEGNet import EEGNet_torch
from operations import prepare_data
from tools import use_wandb
import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm
import wandb



def train():
    pass
    #return

if __name__ == "__main__":
    use_wb = 1
    DEVICE = torch.device('cpu')
    num_epochs = 100
    
    if use_wb == 1:
        use_wandb(projectname="new-test")

    #Prepare Data
    test_dataloader, trainloader = prepare_data()
    
    net = EEGNet_torch()
    
    criterion = nn.CrossEntropyLoss()
    #nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    
    #Train the Network

    
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        
        nb_tr_steps = 0 
        running_loss = 0.0
        
        for step, batch in enumerate(tqdm(trainloader, desc="Iteration"),0):
            batch = tuple(t.to(DEVICE) for t in batch)
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = batch
            
            input_ = torch.zeros([1,2,129,2500])
            input_[0] = inputs
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = net(input_)
            loss = criterion(outputs[0], labels)
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
            nb_tr_steps += 1
            
            if step % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {step + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
             
        loss_ = running_loss / nb_tr_steps
        wandb.log({"loss":loss_})
            
            
print('Finished Training')