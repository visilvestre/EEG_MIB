#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 14:54:15 2022

@author: vlourenco
"""

import numpy as np
import torch
from torch import nn as nn
from tqdm import tqdm
import wandb
from torch.nn import MSELoss
import argparse
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--gradient_accumulation_step", type=int, default=1)

args = parser.parse_args()
verbose = 1

DEVICE = torch.device("mps")
file = '/Users/vlourenco/Documents/GitHub/EEG_MIB/EEGNet Translated to PyTorch/P1_a1.mat'


class TrainDataset(Dataset):
    
    def __init__(self, training_data): 
        x = training_data[0]
        y = training_data[1]
        self.len = x.shape[0]
        self.x_data = torch.from_numpy(x).type(torch.float32)
        self.y_data = torch.from_numpy(y).type(torch.float32)
        
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
        
    def __len__(self):
        return self.len



def test_score_model(model: nn.Module, test_dataloader: DataLoader, use_zero=False):

    preds, y_test = test_epoch(model, test_dataloader)
    non_zeros = np.array(
        [i for i, e in enumerate(y_test) if e != 0 or use_zero])

    test_preds_a7 = np.clip(preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(y_test, a_min=-3., a_max=3.)
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)


    preds = preds[non_zeros]
    y_test = y_test[non_zeros]



    mae = np.mean(np.absolute(preds - y_test))
    corr = np.corrcoef(preds, y_test)[0][1]

    preds = preds >= 0
    y_test = y_test >= 0

    f_score = f1_score(y_test, preds, average="weighted")
    acc = accuracy_score(y_test, preds)


    return acc, mae, corr, f_score, mult_a7



##Used for testing purposes, not in train()
def test_epoch(model: nn.Module, test_dataloader: DataLoader):
    """
    model().eval()
    
    preds = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch = tuple(t.to(DEVICE) for t in batch)

            input_t, label_t = batch
            
            output = model.test() #### CRITICAL , CONFIGURE HERE

            logits = output

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.detach().cpu().numpy()

            logits = np.squeeze(logits).tolist()
            label_ids = np.squeeze(label_ids).tolist()

            preds.extend(logits)
            labels.extend(label_t)

        preds = np.array(preds)
        labels = np.array(labels)

    return preds, labels
    """

    model.eval()
    
    preds = []
    labels = []
    DEVICE = torch.device("cpu")
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch = tuple(t.to(DEVICE) for t in batch)
    
            input_t, label_t = batch
            input_ = torch.zeros([1,2,129,2500])
            input_[0] = input_t
            
            output = model.test() #### CRITICAL , CONFIGURE HERE
            
            logits = output
    
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.detach().cpu().numpy()
    
            logits = np.squeeze(logits).tolist()
            label_ids = np.squeeze(label_ids).tolist()
    
            preds.extend(logits)
            labels.extend(label_t)
    
        preds = np.array(preds)
        labels = np.array(labels)
    
    return preds, labels


#Needed for train()
def train_epoch(model: nn.Module, train_dataloader: DataLoader):
    #setup model to train
    model().train(mode=True)
    
    """
    #Original_model
    #Variables declaration
    tr_loss = 0
    nb_tr_examples = 0
    nb_tr_steps = 0
    
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(DEVICE) for t in batch)
        
        input_t, labels_t = batch
        
        output_t = model() #### NEED TO UPDATE HERE WITH MODEL STRUCTURE....
        
        logits = output_t
        
        loss_fct = MSELoss()
        loss = loss_fct(logits.view(-1), labels_t.view(-1))
        
        if args.gradient_accumulation_step > 1:
            loss = loss / args.gradient_accumulation_step
       
        
        tr_loss += loss.item()
        nb_tr_steps += 1
        
    return tr_loss / nb_tr_steps
    """

    #Variables Declaration
    tr_loss = 0
    nb_tr_examples = 0
    nb_tr_steps = 0
    
    DEVICE = torch.device("cpu") #Define "cpu" or "mds" ("mps" if torch.backends.mps.is_available() else "cpu")
    
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(DEVICE) for t in batch)
        
        input_t, labels_t = batch
        input_ = torch.zeros([1,2,129,2500])
        input_[0] = input_t
        
        output_t = model().forward(x=input_) #### NEED TO UPDATE HERE WITH MODEL STRUCTURE....
        
        logits = output_t
        
        loss_fct = nn.CrossEntropyLoss() ##Changed from MSE to CrossEntropy
        #MSELoss()
        loss = loss_fct(logits.view(-1), labels_t.view(-1))
        
        
        if args.gradient_accumulation_step > 1:
            loss = loss / args.gradient_accumulation_step
            tr_loss += loss.item()
            nb_tr_steps += 1
        
        tr_loss += loss.item()
        nb_tr_steps += 1
        
    return tr_loss / nb_tr_steps


#Needed for train()
def eval_epoch(model: nn.Module, dev_dataloader: DataLoader):
    #Set model to evaluation
    model.eval()
    
    #Declare variables
    dev_loss = 0
    nb_dev_examples = 0
    nb_dev_steps = 0
    
    #Declare no_grad for economy of memory since we don't need .backward()
    with torch.no_grad():
        #for each batch, enumerate a step
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
            #set the batch to a DEVICE
            batch = tuple(t.to(DEVICE) for t in batch)
            
            input_e, label_e = batch
            output_e = model.test() ####### CRITICAL! CONFIGURE MODEL 
            
            logits = output_e
            
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), label_e.view(-1))
            
            if args.gradient_accumulation_step > 1:
                loss = loss / args.gradient_accumulation_step
                
            dev_loss += loss.item()
            nb_dev_steps += 1
            
    return dev_loss / nb_dev_steps
    
    

def train(model, train_dataloader, validation_dataloader, test_dataloader, verbose = 0):
    valid_losses = []
    test_accuracies = []
    best_loss = 10
    
    for epoch_i in range(int(args.n_epochs)):
        train_loss = train_epoch(model, train_dataloader)
        valid_loss = eval_epoch(model, validation_dataloader)
        test_acc, test_mae, test_corr, test_f_score, test_acc7 = test_score_model(model, test_dataloader)
        
        if verbose == 1:
            print("epoch:{}, train_loss:{:.4f}, valid_loss:{:.4f}, test_acc:{:.4f}".format(
                epoch_i, train_loss, valid_loss, test_acc))
            print("current mae:{:.4f}, current acc:{:.4f}, acc7:{:.4f}, f1:{:.4f}, corr:{:.4f}".format(
                test_mae, test_acc, test_acc7, test_f_score, test_corr))
        
        valid_losses.append(valid_loss)
        test_accuracies.append(test_acc)
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_acc = test_acc
            best_mae = test_mae
            best_corr = test_corr
            best_f_score = test_f_score
            best_acc_7 = test_acc7
        
        if verbose == 1:
            print("best mae:{:.4f}, acc:{:.4f}, acc7:{:.4f}, f1:{:.4f}, corr:{:.4f}".format(
                best_mae, best_acc, best_acc_7, best_f_score, best_corr))
        
        wandb.log(
            (
                {
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "test_acc": test_acc,
                    "test_mae": test_mae,
                    "test_corr": test_corr,
                    "test_f_score": test_f_score,
                    "test_acc7": test_acc7,
                    "best_valid_loss": min(valid_losses),
                    "best_test_acc": max(test_accuracies),
                }
            )
        )