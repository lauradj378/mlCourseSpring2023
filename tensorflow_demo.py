#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
import dl.networks
from sklearn.model_selection import ParameterGrid
import torch.utils.tensorboard

"""
all of the parameters and losses are arbitrary in this code
this is just to show how tensorboard works
"""
params = {
        'r': [-1, -2],
        'c': [0.7, 1.1],
}
params = list(ParameterGrid(params)) #all combinations of the parameter elements

def run(params):
    
    c = params['c']
    r = params['r']
    
    dirname = f'r{r}_c{c}'
    writer = SummaryWriter(log_dir=dirname)
    #print(dirname)
    
    for epoch in range(1,20):
        train_loss = c * epoch**r
        test_loss = 1.1 * c * epoch**r
        writer.add_scalar('train_loss', train_loss, epoch) #stores the training loss we computed
        writer.add_scalar('test_loss', test_loss, epoch)
        
    writer.close()

        
    x = np.linspace(-5,5,100)
    fig, ax = plt.subplots()
    ax.plot(x,np.sin(x))

if __name__ == "__main__":
    for p in params:
        run(p)
        
        
"""
Command - tensorboard --log_dir runs/
writer.close - 

"""