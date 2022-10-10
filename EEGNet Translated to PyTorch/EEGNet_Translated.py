# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import scipy
import matplotlib.pyplot as plt
import numpy as np
import torch
import sklearn
from torch import flatten
from torch import nn as nn
from tqdm import tqdm
import wandb
from torch.nn import MSELoss
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
from torch.utils.data import Dataset, DataLoader
import argparse
from sklearn.metrics import accuracy_score, f1_score


parser = argparse.ArgumentParser()

parser.add_argument("--gradient_accumulation_step", type=int, default=1)

args = parser.parse_args()
verbose = 1

#from global_configs import DEVICE

DEVICE = torch.device("mps")
file = '/Users/vlourenco/Documents/GitHub/EEG_MIB/EEGNet Translated to PyTorch/P1_a1.mat'

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


#### EEGNet in Tensorflow ####
def EEGNet(nb_classes, Chans = 64, Samples = 128, 
             dropoutRate = 0.5, kernLength = 64, F1 = 8, 
             D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout'):
    """ Keras Implementation of EEGNet
    http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta
    Note that this implements the newest version of EEGNet and NOT the earlier
    version (version v1 and v2 on arxiv). We strongly recommend using this
    architecture as it performs much better and has nicer properties than
    our earlier version. For example:
        
        1. Depthwise Convolutions to learn spatial filters within a 
        temporal convolution. The use of the depth_multiplier option maps 
        exactly to the number of spatial filters learned within a temporal
        filter. This matches the setup of algorithms like FBCSP which learn 
        spatial filters within each filter in a filter-bank. This also limits 
        the number of free parameters to fit when compared to a fully-connected
        convolution. 
        
        2. Separable Convolutions to learn how to optimally combine spatial
        filters across temporal bands. Separable Convolutions are Depthwise
        Convolutions followed by (1x1) Pointwise Convolutions. 
        
    
    While the original paper used Dropout, we found that SpatialDropout2D 
    sometimes produced slightly better results for classification of ERP 
    signals. However, SpatialDropout2D significantly reduced performance 
    on the Oscillatory dataset (SMR, BCI-IV Dataset 2A). We recommend using
    the default Dropout in most cases.
        
    Assumes the input signal is sampled at 128Hz. If you want to use this model
    for any other sampling rate you will need to modify the lengths of temporal
    kernels and average pooling size in blocks 1 and 2 as needed (double the 
    kernel lengths for double the sampling rate, etc). Note that we haven't 
    tested the model performance with this rule so this may not work well. 
    
    The model with default parameters gives the EEGNet-8,2 model as discussed
    in the paper. This model should do pretty well in general, although it is
	advised to do some model searching to get optimal performance on your
	particular dataset.
    We set F2 = F1 * D (number of input filters = number of output filters) for
    the SeparableConv2D layer. We haven't extensively tested other values of this
    parameter (say, F2 < F1 * D for compressed learning, and F2 > F1 * D for
    overcomplete). We believe the main parameters to focus on are F1 and D. 
    Inputs:
        
      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer. We found
                        that setting this to be half the sampling rate worked
                        well in practice. For the SMR dataset in particular
                        since the data was high-passed at 4Hz we used a kernel
                        length of 32.     
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 8, F2 = F1 * D. 
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.
    """
    
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = Input(shape = (Chans, Samples, 1))

    ##################################################################
    block1       = Conv2D(F1, (1, kernLength), padding = 'same',
                                   input_shape = (Chans, Samples, 1),
                                   use_bias = False)(input1)
    block1       = BatchNormalization()(block1)
    block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    block1       = BatchNormalization()(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((1, 4))(block1)
    block1       = dropoutType(dropoutRate)(block1)
    
    block2       = SeparableConv2D(F2, (1, 16),
                                   use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization()(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D((1, 8))(block2)
    block2       = dropoutType(dropoutRate)(block2)
        
    flatten      = Flatten(name = 'flatten')(block2)
    
    dense        = Dense(nb_classes, name = 'dense', 
                         kernel_constraint = max_norm(norm_rate))(flatten)
    softmax      = Activation('softmax', name = 'softmax')(dense)
    
    return Model(inputs=input1, outputs=softmax)


def main_tensorflow():
    # Data Preparation
    file = '/Users/vlourenco/Documents/GitHub/EEG_MIB/EEGNet Translated to PyTorch/P1_a1.mat'
    mat                     = loadeeg(file)
    #x, mat_framed          = framedata(mat, verbose = 1)
    d, j                    = findindexes(mat)
    deviant                 = getdeviants(mat, d)
    t,k                     = findindexes(mat, triggers_list=[25,65,45])
    tonic                   = gettonics(mat, t)
    x, y                    = distributedata(tonic, deviant)
    x_shuffled, y_shuffled  = datashuffler(x,y)
    y_nn                    = prepareForCrossEntropy(y_shuffled)
    
    #EEGNet Model
    model                   = EEGNet(2, Chans = 129, Samples = 2500, dropoutRate = 0.5, kernLength = 500, F1 = 8, D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout')
    model.compile(optimizer="Adam", loss = "BinaryCrossentropy", metrics = ['accuracy'])
    model.fit(x_shuffled[0:218],y_nn[0:218],epochs = 5)
    model.evaluate(x_shuffled[218:],y_nn[218:])


############################### Pytorch coding ###############################


class TrainDataset(Dataset):
    
    def __init__(self, training_data): 
        xy = training_data
        x = training_data[0]
        y = training_data[1]
        self.len = x.shape[0]
        self.x_data = torch.from_numpy(x).type(torch.float32)
        self.y_data = torch.from_numpy(y).type(torch.float32)
        
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
        
    def __len__(self):
        return self.len



""" Implementation of SeparableConv2d made available in Tensorflow, not necessarily correct..
https://stackoverflow.com/questions/65154182/implement-separableconv2d-in-pytorch
""" 

class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   groups=in_channels, bias=bias, padding=1)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 
                                   kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
    
""" Another implementation from Github 
https://gist.github.com/bdsaglam/84b1e1ba848381848ac0a308bfe0d84c"""

class SeparableConv2d(torch.nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 padding_mode='zeros',
                 depth_multiplier=1,
        ):
        super().__init__()
        
        intermediate_channels = in_channels * depth_multiplier
        self.spatialConv = torch.nn.Conv2d(
             in_channels=in_channels,
             out_channels=intermediate_channels,
             kernel_size=kernel_size,
             stride=stride,
             padding=padding,
             dilation=dilation,
             groups=in_channels,
             bias=bias,
             padding_mode=padding_mode
        )
        self.pointConv = torch.nn.Conv2d(
             in_channels=intermediate_channels,
             out_channels=out_channels,
             kernel_size=1,
             stride=1,
             padding=0,
             dilation=1,
             bias=bias,
             padding_mode=padding_mode,
        )
    
    def forward(self, x):
        return self.pointConv(self.spatialConv(x))


def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

class EEGNet_torch(torch.nn.Module):
    #EEGNet() Model in pytorch
    
    def __init__(self,
                 in_channels = 2,
                 F1 = 8,
                 kernLength = 500,
                 device_type = "cpu",
                 dropoutRate = 0.5,
                 depth_multiplier = 1,
                 nb_classes = 2
                ):
        super(EEGNet_torch, self).__init__()
        """
        #Block1
        self.b1step1 = nn.Conv2d(in_channels = 2, 
                                 out_channels = 8, 
                                 kernel_size = (1, kernLength), 
                                 padding = 'same',
                                 dilation= 1,
                                 groups = 2,
                                 bias = False,
                                 padding_mode = 'zeros',
                                 device = None,
                                 dtype = None)
        self.b1step2 = nn.BatchNorm2d(8)
        self.b1step3 = nn.Conv2d(in_channels = 8, 
                                 out_channels = 8, 
                                 kernel_size = (1, kernLength), 
                                 stride = (1,1), 
                                 padding = 'same',
                                 dilation = 1,
                                 groups= 8,
                                 bias = False, 
                                 padding_mode = 'zeros', 
                                 device = None, 
                                 dtype = None)
        self.b1step4 = nn.BatchNorm2d(8)
        self.b1step5 = nn.ELU(alpha=1.0, 
                              inplace=False)
        self.b1step6 = nn.AvgPool2d((1,4), 
                                    stride=None, 
                                    padding=0, 
                                    ceil_mode=False, 
                                    count_include_pad=True, 
                                    divisor_override=None)
        self.b1step7 = nn.Dropout(dropoutRate)

        #Block2
        
        intermediate_channels = 8 * depth_multiplier
        self.spatialConv = torch.nn.Conv2d(
             in_channels= 8,
             out_channels= intermediate_channels,
             kernel_size=3,
             stride=1,
             padding=0,
             dilation=1,
             groups=8,
             bias=False,
             padding_mode='zeros'
        )
        self.pointConv = torch.nn.Conv2d(
             in_channels= intermediate_channels,
             out_channels= 8,
             kernel_size=1,
             stride=1,
             padding=0,
             dilation=1,
             bias=False,
             padding_mode='zeros',
        ) 
        #self.b2step1 = SeparableConv2d(in_channels = 8 , out_channels = 8)
        self.b2step2 = nn.BatchNorm2d(num_features = 8)
        self.b2step3 = nn.ELU(alpha=1.0, 
                              inplace=False)
        self.b2step4 = nn.AvgPool2d((1,8), 
                                    stride=None, 
                                    padding=0, 
                                    ceil_mode=False, 
                                    count_include_pad=True, 
                                    divisor_override=None)
        
        self.b2step5 = nn.Dropout(dropoutRate)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(in_features = 78232,
                               out_features = 2, 
                               bias=True, 
                               device=None, 
                               dtype=None)
        self.softmax = nn.Softmax(dim=None)
        """
        intermediate_channels = 8 * depth_multiplier
        self.sequential = nn.Sequential(nn.Conv2d(in_channels = 2, 
                                                  out_channels = 8, 
                                                  kernel_size = (1, kernLength), 
                                                  padding = 'same',
                                                  dilation= 1,
                                                  groups = 2,
                                                  bias = False,
                                                  padding_mode = 'zeros',
                                                  device = None,
                                                  dtype = None), 
                                        nn.BatchNorm2d(8), 
                                        nn.Conv2d(in_channels = 8, 
                                                  out_channels = 8, 
                                                  kernel_size = (1, kernLength), 
                                                  stride = (1,1), 
                                                  padding = 'same',
                                                  dilation = 1,
                                                  groups= 8,
                                                  bias = False, 
                                                  padding_mode = 'zeros', 
                                                  device = None, 
                                                  dtype = None),
                                        nn.BatchNorm2d(8), 
                                        nn.ELU(alpha=1.0, 
                                               inplace=False), 
                                        nn.AvgPool2d((1,4), 
                                                     stride=None, 
                                                     padding=0, 
                                                     ceil_mode=False, 
                                                     count_include_pad=True, 
                                                     divisor_override=None),
                                        nn.Dropout(dropoutRate), 
                                        torch.nn.Conv2d(in_channels= 8,
                                                        out_channels= intermediate_channels,
                                                        kernel_size=3,
                                                        stride=1,
                                                        padding=0,
                                                        dilation=1,
                                                        groups=8,
                                                        bias=False,
                                                        padding_mode='zeros'), 
                                        torch.nn.Conv2d(in_channels= intermediate_channels,
                                                        out_channels= 8,
                                                        kernel_size=1,
                                                        stride=1,
                                                        padding=0,
                                                        dilation=1,
                                                        bias=False,
                                                        padding_mode='zeros'), 
                                        nn.BatchNorm2d(num_features = 8), 
                                        nn.ELU(alpha=1.0, inplace=False), 
                                        nn.AvgPool2d((1,8), 
                                                     stride=None, 
                                                     padding=0, 
                                                     ceil_mode=False, 
                                                     count_include_pad=True, 
                                                     divisor_override=None), 
                                        nn.Dropout(dropoutRate), 
                                        nn.Flatten(), 
                                        nn.Linear(in_features = 78232,
                                                  out_features = 2, 
                                                  bias=True, 
                                                  device=None, 
                                                  dtype=None),
                                        nn.Softmax(dim=None) )
        
    
    def forward(self, x):
        """
        #Block1
        block1 = self.b1step1(x)  
        block1 = self.b1step2(block1)
        block1 = self.b1step3(block1)
        block1 = self.b1step4(block1)
        block1 = self.b1step5(block1)
        block1 = self.b1step6(block1)
        block1 = self.b1step7(block1)


        block2 = self.pointConv(self.spatialConv(block1))
        block2 = self.b2step2(block2)
        block2 = self.b2step3(block2)
        block2 = self.b2step4(block2)
        block2 = self.b2step5(block2)
        #block2 = self.flatten(block2)

        flatten = self.flatten(block2)

        dense = self.dense(flatten)

        softmax = self.softmax(dense)
        """
        
        softmax = self.sequential(x)
        return softmax

#used for scoring
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
        



def main_torch():
    
    # Data Preparation
    mat                     = loadeeg(file, verbose = 0)
    #x, mat_framed           = framedata(mat, verbose = 1)
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
    
    ### test zone ###

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
    main_tensorflow()
    #main_torch()