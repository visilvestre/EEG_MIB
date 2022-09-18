#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 14:41:38 2022

@author: vlourenco
"""
import scipy
import matplotlib.pyplot as plt
import numpy as np


#### Data Engineering Functions ####
def loadeeg(file, verbose = 0):
    #Load EEG Data
    mat = scipy.io.loadmat(file)
    
    #Tests to comprehend the data
    if verbose == 1:
        print(f'print mat type: {type(mat)}')
        for a in mat:
            print(f'Key in mat: {a}')
        
        print(' ')
        shape = np.shape(mat['xContinuous'])
        triggers = mat['triggers']
        onsets = mat['onsets']
        
        print(f'mat xContinuous shape: {shape}')
        print(f'mat triggers: {triggers}')
        print(f'mat onsets: {onsets}')
        
        n = 1209
        subtract = mat['onsets'][n]-mat['onsets'][n-1]
        print(f'space between two onsets: {subtract}')
        print(' ')
        
        x = np.arange(0, 10000, 1)
        plt.plot(x, mat['xContinuous'][0,10000:20000])
        
    return mat


def framedata(mat, start=119, end =124, verbose = 0):
    '''Framing data between two onsets'''
    # Get start and end of the onsets
    st = mat['onsets'][start]
    en = mat['onsets'][end]
    
    # Create an x axis
    x = np.arange(st, en, 1)
    
    #framed mat
    mat_framed = mat['xContinuous'][0,st[0]:en[0]]
    
    # Plot the data between onsets
    if verbose == 1:
        plt.plot(x, mat_framed)

    return x, mat_framed

def findindexes(mat, triggers_list = [30,35,40,50,55,60,70,75,80], verbose = 0):
    #Find indexes d where the number is in the list ( list of deviant triggers ) 
    d,j = np.where(mat['triggers'] == triggers_list)
    
    if verbose == 1:
        print(d)
    
    return d, j
    

def getdeviants(mat, d, verbose = 0):    
    '''Get chunks of data from deviant trials'''
    deviant = []
    
    for n in d:
        start = mat['onsets'][n] #start 4 chords before
        if n == 1214:
            end = mat['onsets'][1214] + 625
        else:
            end = mat['onsets'][n+1]   #end when the next onset starts
            deviant.append(mat['xContinuous'][:,start[0]:end[0]])
            
    #Used to print when only one channel was selected..
    #a = len(deviant)
    #i = 0
    #while i < a:
    #    x = np.arange(0, np.shape(deviant[i])[0], 1)
    #    plt.plot(x, deviant[i])
    #    i+=1
    
    if verbose == 1: 
        i = 0
        for d in deviant:
            print(f'Deviant block {i} Shape: {np.shape(d)}')
            i += 1
            
    return deviant
    
    
def gettonics(mat, t, verbose = 0):
    
    tonic = []
    
    for n in t:
        start = mat['onsets'][n] #start 4 chords before
        if n == 1214:
            end = mat['onsets'][1214] + 625
        else:
            end = mat['onsets'][n+1]   #end when the next onset starts
        tonic.append(mat['xContinuous'][:,start[0]:end[0]])
    
    #Used to print when only one channel was selected..
    #a = len(tonic)
    #i = 0
    #while i < a:
    #    x = np.arange(0, np.shape(tonic[i])[0], 1)
    #    plt.plot(x, tonic[i])
    #    i+=1
    if verbose == 1: 
        i = 0
        for d in tonic:
            print(f'Tonic block {i} Shape: {np.shape(d)}')
            i += 1

    return tonic

def distributedata(tonic, deviant, verbose = 0):

    #Distribute data into X
    x = np.zeros((len(tonic)+len(deviant),129,2500))
    
    if verbose == 1:
        print(np.shape(x))
    
    
    i = 0
    for t in tonic:
        x[i,:,0:2500] = t[:,0:2500]
        i += 1


    for d in deviant:
        x[i,:,0:2500] = d[:,0:2500]
        i  += 1

    #Create labels and fulfill
    y = np.zeros(len(tonic)+len(deviant))
    y[0:len(tonic)] = np.ones(len(tonic))
    
    if verbose == 1: 
        print(f'y vector: {y}')
        print(f'x shape: {np.shape(x)}')
    return x, y

def datashuffler(x, y):
    #Shuffle Data in numpy arrays
    shuffler = np.random.permutation(len(x))
    x_shuffled = x[shuffler]
    y_shuffled = y[shuffler]
    
    return x_shuffled, y_shuffled

def prepareForCrossEntropy(y_shuffled, verbose = 0):
    # Organize Label for Neural Network Cross Entropy validation
    
    y_size = np.shape(y_shuffled)[0]
    y_nn = np.zeros((y_size,2))
    
    if verbose == 1:
        print(f'y_shuffled shape: {np.shape(y_shuffled)}')
        print(f'y_nn shape: {np.shape(y_nn)}')
    
    i = 0
    while i < np.shape(y_shuffled)[0]:
        if y_shuffled[i] == 1:
            y_nn[i][0] = 1
        else:
            y_nn[i][1] = 1
        i += 1
    
    return y_nn

def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))