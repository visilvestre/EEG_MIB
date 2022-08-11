#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 16:32:42 2022

@author: viniciussilvestrelourenco
"""
import os
import mne

#Crawler to find files
##Read the files and add to an eegraw vector of EEGLAB objects
class pp:

    def read_eeg(dataset_path):
        """
            Method to read the eeg from a folder, 
            the OS liberary do a crawling on the selected
            folder while mne.io.read_raw_eeglab reads the 
            egg in eeglab format
            
            Args:
                dataset_path = String with dataset path for the images
            
            Return:
                images = List with eegs in eeglab raw format
        """
        eegraw = []
        filename = []
        for path, subdir, files in os.walk(dataset_path):
                for file in files:
                    if file[-3:] == "set":
                        raw = mne.io.read_raw_eeglab(f"{path}/{file}")
                        eegraw.append(raw)
                        filename.append(file)
        
        return eegraw, filename

#raw = mne.io.read_raw_eeglab('/Users/viniciussilvestrelourenco/Desktop/eegexperiment/sub-001/eeg/sub-001_task-Experiment_eeg.set', preload=True)


