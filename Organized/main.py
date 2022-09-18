#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 15:17:13 2022

@author: vlourenco
"""

import wandb
from EEGNet import EEGNet_torch
from operations import train_epoch, prepare_data
from tools import use_wandb


def main():
    """
    main method is responsible for running the process logic flow 
    to 
    1- Prepare the data
    2- Test EEGNet
    ...
    """
    
    use_wandb(projectname="my-test-project")
    test_dataloader, train_dataloader = prepare_data()

    
    ## Testing EEGNet
    model = EEGNet_torch
    print(model)
    
    n_epochs = 5
    
    ###### test zone ######

    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_dataloader)
        print(f'train_loss: {train_loss}')
        
        #Log to wandb website
        wandb.log({"loss": train_loss})
        
        #other tests
        #print(train_loss.backward())
    
    
    #acc, mae, corr, f_score, mult_a7 = test_score_model(model, test_dataloader)
    #print(f"acc: {acc},  mae: {mae}, corr: {corr}, f_score: {f_score}, mult_a7: {mult_a7}")
    #model = EEGNet_torch(train_x)
    #a = model.forward()
    #print(a)
    
    
if __name__ == "__main__":
    main()
