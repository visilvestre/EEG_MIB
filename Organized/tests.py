#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 15:17:13 2022

@author: vlourenco
"""

import wandb
from EEGNet import EEGNet_torch
from operations import train_epoch_test, prepare_data
from tools import use_wandb

use_wandb = 0

if use_wandb == 1:
    use_wandb(projectname="my-test-project")

def main():
    """
    main method is responsible for running the logic
    1- Prepare the data
    2- Test EEGNet
    ...
    """
    #Prepare Data
    test_dataloader, train_dataloader = prepare_data()

    
    #Testing EEGNet
    model = EEGNet_torch
    print(model)
    
    n_epochs = 1
    
    ###### test zone ######

    for epoch in range(n_epochs):
        train_loss, output_t = train_epoch_test(model, train_dataloader)
        print(f'train_loss: {train_loss}')
        
        #Log to wandb website
        if use_wandb == 1:
            wandb.log({"loss": train_loss})
        
        #other tests
        print(f"output_dtype:{output_t.dtype}, size:{output_t.size()}")
        output_t.sum().backward()
    
    
    #acc, mae, corr, f_score, mult_a7 = test_score_model(model, test_dataloader)
    #print(f"acc: {acc},  mae: {mae}, corr: {corr}, f_score: {f_score}, mult_a7: {mult_a7}")
    #model = EEGNet_torch(train_x)
    #a = model.forward()
    #print(a)
    
    
if __name__ == "__main__":
    main()
