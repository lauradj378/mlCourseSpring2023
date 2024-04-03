#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.append("/Users/laurahull/Documents/UCF/Spring_23/Machine_Learning_Course/dl")

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.models as models
import dl.models
import dl.callbacks
from dl.networks import LeNet5Modified
from torchvision.transforms import Lambda
from tensorboard.plugins.hparams import api as hp
import time
import platform
import psutil
import torch.nn.init as init

logdir = '/Users/laurahull/Documents/UCF/Spring_23/Machine_Learning_Course/dl/assignments/runs'

params = {
    'network': LeNet5Modified,
    'learning_rate': 1e-2,
    'batch_size': 64,
    'epochs': 40,
    #'epochs': 2,
    'n_classes': 10,
    'normalize': 'True',
    'dropout': 0.5,
    'optimizer': [torch.optim.SGD, torch.optim.Adagrad, torch.optim.Adam]}

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size = params['batch_size'],shuffle = True)
test_dataloader = DataLoader(test_data, batch_size = params['batch_size'],shuffle = True)

optimizer_val = params['optimizer']

if __name__ == '__main__':
    start_time = time.perf_counter()
    for jj in range(len(optimizer_val)): ##
        network = params['network'](params['n_classes'],params['normalize'],params['dropout']) ##
        optimizer = optimizer_val[jj](network.parameters(), lr = params['learning_rate'])
        model = dl.models.Model(network,optimizer,F.cross_entropy,params['epochs'])
        log_dir = f'/Users/laurahull/Documents/UCF/Spring_23/Machine_Learning_Course/dl/assignments/runs/optimizer={optimizer_val[jj]}' ##
        callbacks = [dl.callbacks.PrintLoss(),
                    dl.callbacks.PrintMeanTestLoss(),
                    dl.callbacks.PrintTestCorrect(),
                    dl.callbacks.GDLoss(log_dir,optimizer_val[jj])] ##
        model.fit(train_dataloader,test_dataloader,callbacks)
    end_time = time.perf_counter()
    total_time = end_time - start_time
    cpu = platform.processor()
    memory = psutil.virtual_memory().total
    if platform.system() == 'Darwin':
        os = 'macOS'
    else:
        os = platform.linux_distribution()
    print(f"Total runtime: {total_time:.2f} seconds")
    print(f"CPU: {cpu}")
    print(f"Memory: {memory} bytes")
    print(f"Operating system: {os}")   