# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import scipy
import matplotlib.pyplot as plt
import numpy as np
#import EEGNet
import torch
from torch import nn as nn
from tqdm import tqdm
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

file = '/Users/vlourenco/Documents/GitHub/EEG_MIB/EEGNet Translated to PyTorch/P1_a1.mat'

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

def PrepareForCrossEntropy(y_shuffled, verbose = 0):
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
    y_nn                    = PrepareForCrossEntropy(y_shuffled)
    
    #EEGNet Model
    model                   = EEGNet(2, Chans = 129, Samples = 2500, dropoutRate = 0.5, kernLength = 500, F1 = 8, D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout')
    model.compile(optimizer="Adam", loss = "BinaryCrossentropy", metrics = ['accuracy'])
    model.fit(x_shuffled[0:218],y_nn[0:218],epochs = 5)
    model.evaluate(x_shuffled[218:],y_nn[218:])


############################### Tensorflow to Pytorch ###############################


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


class EEGNet_torch:
    #EEGNet() Model in pytorch
    ## dependent of the class SeparableConv2d
    
    def __init__(self, train_x, F1 = 8, kernLength = 500, nb_classes =2, dropoutRate = 0.5, device = torch.device('mps')):
        self.train_x = train_x
        self.F1 = F1
        self.kernLength = kernLength
        self.nb_classes = nb_classes
        self.dropoutRate = dropoutRate
        self.device = device
    
    def forward(self):
        #Block1
        block1 = nn.Conv2d(in_channels = 218 , out_channels = self.F1, kernel_size = (1, self.kernLength), padding = 'same',
                           dilation= 1,groups = 1,bias = False,padding_mode = 'zeros',device = device,dtype = None)(self.train_x)
        
        block1 = nn.BatchNorm2d(8)(block1)
        block1 = nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = (1, self.kernLength), stride = (1,1), padding = 'same',dilation = 1,
                           groups= 8, bias = False, padding_mode = 'zeros', device = device, dtype = None)(block1)
        block1 = nn.BatchNorm2d(8)(block1)
        block1 = nn.ELU(alpha=1.0, inplace=False)(block1)
        block1 = nn.AvgPool2d((1,4), stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)(block1)
        block1 = nn.Dropout(self.dropoutRate)(block1)
        
        
        #Block2
        block2 = SeparableConv2d(in_channels = 8 , out_channels = 8 )(block1)
        block2 = nn.BatchNorm2d(num_features = 8)(block2)
        block2 = nn.ELU(alpha=1.0, inplace=False)(block2)
        block2 = nn.AvgPool2d((1,8), stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)(block2)
        block2 = nn.Dropout(self.dropoutRate)(block2)
        
        flatten = torch.flatten(block2)
        
        dense = nn.Linear(in_features = flatten.size()[0], out_features = self.nb_classes, bias=True, device=device, dtype=None)(flatten)
        
        softmax = nn.Softmax(dim=None)(dense)
        
        return softmax


def main_torch():
    
    # Data Preparation
    mat                     = loadeeg(file)
    #x, mat_framed           = framedata(mat, verbose = 1)
    d, j                    = findindexes(mat)
    deviant                 = getdeviants(mat, d)
    t,k                     = findindexes(mat, triggers_list=[25,65,45])
    tonic                   = gettonics(mat, t)
    x, y                    = distributedata(tonic, deviant)
    x_shuffled, y_shuffled  = datashuffler(x,y)
    y_nn                    = PrepareForCrossEntropy(y_shuffled)

    #Variables Definition
    nb_classes = 2
    Chans = 129
    Samples = 2500
    dropoutRate = 0.5
    kernLength = 500
    F1 = 8
    D = 2
    F2 = 16
    norm_rate = 0.25
    dropoutType = 'Dropout'
    data_x = x_shuffled[0:218]
    
    #Define input tensor
    input1 = torch.from_numpy(data_x)
    
    #Prepare data in the format torch accepts
    train_x = torch.zeros([1,218,129,2500])
    train_x.size()

    train_x[0] = input1[0:218]
    
    model = EEGNet_torch(train_x)
    a = model.forward()
    
    print(a)
    
    
if __name__ == "__main__":
    main_tensorflow()
    #main_torch()

