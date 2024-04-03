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
from dl.networks import LeNet5
#from torchvision.transforms import Lambda

params = {
    'network': LeNet5,
    'learning_rate': 1e-3,
    'batch_size': 64,
    'epochs': 40,
    'n_classes': 10}


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

if __name__ == '__main__':
    network = params['network']()
    optimizer = torch.optim.SGD(network.parameters(), lr = params['learning_rate'])
    model = dl.models.Model(network,optimizer,F.cross_entropy,params['epochs'])
    callbacks = [dl.callbacks.PrintLoss(),
                 dl.callbacks.PrintMeanTestLoss(),
                 dl.callbacks.PrintTestCorrect()]
    model.fit(train_dataloader,test_dataloader,callbacks)
    torch.save(network.state_dict(),'LeNet-5-MNIST.pth')