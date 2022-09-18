#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 14:43:12 2022

@author: vlourenco
"""
import wandb
from EEGNet import EEGNet_torch
from tools import loadeeg, findindexes, getdeviants, gettonics, distributedata, datashuffler, framedata, prepareForCrossEntropy
from torch.utils.data import DataLoader
from operations import TrainDataset, test_score_model, test_epoch, train_epoch, eval_epoch, train

def main():
    """
    main method is responsible for running the process logic flow 
    to 
    1- Prepare the data
    2- Test EEGNet
    ...
    """
    
    #1- Prepare Data
    file = '/Users/vlourenco/Documents/GitHub/EEG_MIB/EEGNet Translated to PyTorch/P1_a1.mat'
    mat                     = loadeeg(file, verbose = 0)
    x, mat_framed           = framedata(mat, verbose = 0)
    d, j                    = findindexes(mat, verbose = 0)
    deviant                 = getdeviants(mat, d, verbose = 0)
    t,k                     = findindexes(mat, triggers_list=[25,65,45], verbose = 0)
    tonic                   = gettonics(mat, t, verbose = 0)
    x, y                    = distributedata(tonic, deviant, verbose = 0)
    x_shuffled, y_shuffled  = datashuffler(x,y)
    y_nn                    = prepareForCrossEntropy(y_shuffled, verbose = 0)
    
    #Define input tensor
    
    #Prepare train data in the format torch accepts
    x_train = x_shuffled[0:218]
    y_train = y_shuffled[0:218]
    

    #Define train Dataset
    training_data = (x_train, y_train)
    train_dataset = TrainDataset(training_data = training_data)
    train_dataloader = DataLoader(dataset = train_dataset, batch_size = 2, shuffle = True, num_workers=0)
    
    
    #Prepare test data in the format torch accepts'
    x_test = x_shuffled[218:]
    y_test = y_shuffled[218:]
    
    #Define test Dataset
    testing_data = (x_test, y_test)
    test_dataset = TrainDataset(training_data = testing_data)
    test_dataloader = DataLoader(dataset = test_dataset, batch_size = 2, shuffle = True, num_workers=0)
    
    
    ## Testing EEGNet
    model = EEGNet_torch
    print(model)
    
    n_epochs = 1
    
    ###### test zone ######

    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_dataloader)
        print(f'train_loss: {train_loss}')
        #print(train_loss.backward())
    
    
    #acc, mae, corr, f_score, mult_a7 = test_score_model(model, test_dataloader)
    #print(f"acc: {acc},  mae: {mae}, corr: {corr}, f_score: {f_score}, mult_a7: {mult_a7}")
    #model = EEGNet_torch(train_x)
    #a = model.forward()
    #print(a)
    
    
if __name__ == "__main__":
    main()