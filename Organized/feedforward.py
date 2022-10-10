#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 17:30:53 2022

@author: vlourenco
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import wandb


#1 - Download Dataset
#2 - Create Dataloader
#3 - build model
#4 - train
#5 - save trained model

BATCH_SIZE = 128
EPOCHS = 50
WANDB = 1
LEARNING_RATE = 0.001

def use_wandb(projectname):
    ###Configure Wandb to send data and create result graphs
    wandb.init(project = projectname)
    wandb.config = {
      "learning_rate": 0.001,
      "epochs": 50,
      "batch_size": 128
    }
    
def download_mnist_datasets():
    train_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    validation_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    return train_data, validation_data
    

class FeedForwardNet(nn.Module):
    #EEGNet() Model in pytorch
    
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
                                          nn.Linear(28*28, 256),
                                          nn.ReLU(),
                                          nn.Linear(256,10)
                                          )
        self.softmax = nn.Softmax(dim=1)
        
    
    def forward(self, input_data):
        flattened_data = self.flatten(input_data)
        logits = self.dense_layers(flattened_data)
        predictions = self.softmax(logits)
        return predictions
    
def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader
    
def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    steps = 0
    running_loss = 0
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
        a = loss.item()
        wandb.log({"loss_item":a})
        
        running_loss += loss.item()
        steps += 1
    
    loss_avg = running_loss / steps
    print(f"loss: {loss.item()}")
    wandb.log({"loss_avg":loss_avg})
    
    
    
    
    
def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")
    print("Finished training")


if __name__ == "__main__":
    device = torch.device("mps")
    
    #use wandb to plot loss
    if WANDB == 1:
        use_wandb(projectname="audio-test1")
    
    # download data and create data loader
    train_data, _ = download_mnist_datasets()
    train_dataloader = create_data_loader(train_data, BATCH_SIZE)
    
    # construct model and assign it to device
    feed_forward_net = FeedForwardNet().to(device)
    print(feed_forward_net)
    
    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(feed_forward_net.parameters(),
                                 lr=LEARNING_RATE)
    
    # train model
    train(feed_forward_net, train_dataloader, loss_fn, optimiser, device, EPOCHS)

    # save model
    torch.save(feed_forward_net.state_dict(), "feedforwardnet.pth")
    print("Trained feed forward net saved at feedforwardnet.pth")
    
    