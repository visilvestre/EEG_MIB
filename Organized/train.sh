#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 17:16:28 2022

@author: vlourenco
"""

#!/bin/bash

python -u train.py \
--batch_size 8 \
--domain "real" \
--input_size 64 \
--learning_rate 0.01 \
--max_epoch 1 \
--checkpoints_save_dir "./checkpoints/" 