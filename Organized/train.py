#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 16:52:28 2022

@author: vlourenco
"""

from testing import eval_eegnet
from EEGNet import EEGNet_torch_test
from operations import prepare_data
from tools import use_wandb
import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm
import wandb
import os


### PARAMETERS ###
USE_WB = 1
DEVICE_NAME = 'cpu'
DEVICE = torch.device(DEVICE_NAME)
NUM_EPOCHS = 20

#### METHODS ###
def save_model_all(model, save_dir, model_name, epoch):
    """
    :param model:  nn model
    :param save_dir: save model direction
    :param model_name:  model name
    :param epoch:  epoch
    :return:  None
    """
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, model_name)
    save_path = '{}_epoch_{}.pt'.format(save_prefix, epoch)
    print("save all model to {}".format(save_path))
    output = open(save_path, mode="wb")
    torch.save(model.state_dict(), output)
    # torch.save(model.state_dict(), save_path)
    output.close() 

def train(model, NUM_EPOCHS, USE_WB, trainloader, criterion, optimizer):
    #Train the Network
    for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times
        
        nb_tr_steps = 0 
        running_loss = 0.0
        
        for step, batch in enumerate(tqdm(trainloader, desc="Iteration"),0):
            batch = tuple(t.to(DEVICE) for t in batch)
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = batch
            
            #input_ = torch.zeros([2,129,2500,1])
            input_ = torch.zeros([2,129,2500,1])
            input_ = torch.unsqueeze(inputs,dim=3)
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = model(input_)
            loss = criterion(outputs[0], labels)
            loss.backward()
            optimizer.step()
    
            # print statistics
            
            running_loss += loss.item()
            
            #send to wandb
            plot = loss.item()
            if USE_WB == 1:
                wandb.log({"loss_item":plot})
            
            nb_tr_steps += 1
            
            if step % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {step + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
        
        avg_loss = running_loss / nb_tr_steps
        if USE_WB == 1:
            wandb.log({"loss":avg_loss})
    
    return model, optimizer

def train_and_test(model, NUM_EPOCHS, USE_WB, trainloader, test_dataloader, criterion, optimizer,device):
    #Train the Network
    for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times
        
        model.train()
        nb_tr_steps = 0 
        running_loss = 0.0
        
        for step, batch in enumerate(tqdm(trainloader, desc="Iteration"),0):
            batch = tuple(t.to(DEVICE) for t in batch)
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = batch
            
            #input_ = torch.zeros([2,129,2500,1])
            input_ = torch.zeros([2,129,2500,1])
            input_ = torch.unsqueeze(inputs,dim=3)
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = model(input_)
            loss = criterion(outputs[0], labels)
            loss.backward()
            optimizer.step()
    
            # print statistics
            
            running_loss += loss.item()
            
            #send to wandb
            plot = loss.item()
            if USE_WB == 1:
                wandb.log({"loss_item":plot})
            
            nb_tr_steps += 1
            
            if step % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {step + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
        
        avg_loss = running_loss / nb_tr_steps
        accuracy = eval_eegnet(test_dataloader,model,device)
        
        if USE_WB == 1:
            wandb.log({"loss":avg_loss})
            wandb.log({"accuracy":accuracy})
    
    return model, optimizer

### MAIN ###
if __name__ == "__main__":
    
    #Print Parameters
    print(f'USE_WB: {USE_WB},')
    print(f'DEVICE_NAME: {DEVICE_NAME},')
    print(f'NUM_EPOCHS: {NUM_EPOCHS},')
    
    if USE_WB == 1:
        use_wandb(projectname="EEGNet_With_Accuracy")

    #Prepare Data
    test_dataloader, trainloader = prepare_data()
    
    model = EEGNet_torch_test()
    
    criterion = nn.BCELoss()
    #nn.CrossEntropyLoss()
    #nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model, optimizer = train_and_test(model, NUM_EPOCHS, USE_WB, trainloader, test_dataloader, criterion, optimizer, device = DEVICE)
    save_model_all(model,"/Users/vlourenco/Documents/GitHub/EEG_MIB/Organized/Saved_models", "eegnet", NUM_EPOCHS)
            
print('Finished Training')




























