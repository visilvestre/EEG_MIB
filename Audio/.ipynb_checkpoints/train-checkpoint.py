#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 07:23:00 2022

@author: vlourenco
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from urbansounddataset import UrbanSoundDataset
import torchaudio
from cnn import CNNNetwork


BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 0.001
ANNOTATIONS_FILE = "data/UrbanSound8K/metadata/UrbanSound8K.csv"
AUDIO_DIR = "data/UrbanSound8K/audio"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050
USE_GPU = 0    #Several issues when using M1 processor.. 


# class FeedForwardNet(nn.Module):

#     def __init__(self):
#         super().__init__()
#         self.flatten = nn.Flatten()
#         self.dense_layers = nn.Sequential(
#             nn.Linear(28 * 28, 256),
#             nn.ReLU(),
#             nn.Linear(256, 10)
#         )
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, input_data):
#         x = self.flatten(input_data)
#         logits = self.dense_layers(x)
#         predictions = self.softmax(logits)
#         return predictions


# def download_mnist_datasets():
#     train_data = datasets.MNIST(
#         root="data",
#         train=True,
#         download=True,
#         transform=ToTensor(),
#     )
#     validation_data = datasets.MNIST(
#         root="data",
#         train=False,
#         download=True,
#         transform=ToTensor(),
#     )
#     return train_data, validation_data


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
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
        


    #Instatiating our dataset object and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = UrbanSoundDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device = device)
    
    train_dataloader = create_data_loader(usd, BATCH_SIZE)
    
    #construct model and assign to a device
    cnn = CNNNetwork().to(device)
    print(cnn)

    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(cnn, train_dataloader, loss_fn, optimiser, device, EPOCHS)

    # save model
    torch.save(cnn.state_dict(), "cnn_dataset8k.pth")
    print("Trained feed forward net saved at cnn_dataset8k.pth")