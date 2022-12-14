#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 07:23:00 2022

@author: vlourenco
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from stimuliaudiodataset import StimuliAudioDataset
import torchaudio
from cnn import CNNNetwork
import wandb

BATCH_SIZE = 2
EPOCHS = 100
LEARNING_RATE = 0.001
ANNOTATIONS_FILE = "data/StimuliAudio/metadata/stimuli.csv"
AUDIO_DIR = "data/StimuliAudio/audio/"
SAMPLE_RATE = 48000
NUM_SAMPLES = 180000
USE_GPU = 0    #Several issues when using M1 processor.. 
PROJECTNAME = "AudioTraining"
USE_WANDB = 1
SHUFFLE_DATALOADER = False


def use_wandb(projectname, learning_rate, epochs, batch_size):
    ###Configure Wandb to send data and create result graphs
    wandb.init(project = projectname)
    wandb.config = {
      "learning_rate": learning_rate,
      "epochs": epochs,
      "batch_size": batch_size
    }

def create_data_loader(train_data, batch_size, shuff=False):
    train_dataloader = DataLoader(train_data, batch_size=batch_size,shuffle=shuff)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device, wdb):
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    
    if wdb == 1:
        ls = loss.item()
        wandb.log({"Loss": ls})
    print(f"loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, device, epochs, wdb):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device, wdb)
        print("---------------------------")
    print("Finished training")


if __name__ == "__main__":
    if USE_GPU == 1:
        # select a device
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        print(f"Using {device}")
    else:
        device = "cpu"
        print(f"Using {device}")
    
    #Use wandb to log data
    if USE_WANDB == 1:
        use_wandb(projectname=PROJECTNAME, 
                  learning_rate=LEARNING_RATE,
                  epochs=EPOCHS, 
                  batch_size=BATCH_SIZE)


    #Instatiating our dataset object and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=128
    )

    usd = StimuliAudioDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device = device)
    
    train_dataloader = create_data_loader(usd, BATCH_SIZE,SHUFFLE_DATALOADER)
    
    #construct model and assign to a device
    cnn = CNNNetwork().to(device)
    print(cnn)

    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(cnn, train_dataloader, loss_fn, optimiser, device, EPOCHS, USE_WANDB)

    # save model
    torch.save(cnn.state_dict(), "cnn_stimuliaudio.pth")
    print("Trained feed forward net saved at cnn_stimuliaudio.pth")