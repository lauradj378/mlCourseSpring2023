#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from torchvision.transforms import Lambda
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import numpy as np



alexnet = models.alexnet(pretrained=True)
alexnet.eval()

# =============================================================================
# conv1 = network.layer1[0]
# conv2 = network.layer2[0]
# relu1 = network.layer1[1]
# relu2 = network.layer2[1]
# =============================================================================

kernels1 = alexnet.features[0].weight.detach().clone()
#print(kernels1)
kernels2 = alexnet.features[3].weight.detach().clone()
#print(kernels2)

def imshow_grid(img):
    '''
    Input "img" must be converted to a tensor before being called in this function
    '''
    m = img.size(dim=0)
    n = img.size(dim=1)
    height = img.size(dim=2)
    width = img.size(dim=3)
    img = F.pad(img, (1,1,1,1), mode='constant', value=np.nan)
    print(img.size())
    m = img.size(dim=0)
    n = img.size(dim=1)
    height = img.size(dim=2)
    width = img.size(dim=3)
    img = img.reshape([m,height,n,width])
    img = torch.reshape(img,([m*height, n*width]))
    print(img.size())
    plt.figure(figsize=img.size())
    plt.imshow(img)
    plt.axis('off')
#figure1 = plt.figure(figsize=(42,7))
imshow_grid(kernels1)
#figure2 = plt.figure(figsize=(112,42))
imshow_grid(kernels2)

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
image = training_data[0][0]


activation = {}
def get_activation(name):
    def hook(network, input, output):
        activation[name] = output.detach()
    return hook

model = LeNet5()
model.relu1.register_forward_hook(get_activation('relu1'))
model.relu2.register_forward_hook(get_activation('relu2'))

y = image
y.requires_grad_()
#y_grad = y.grad()
#print(y.grad)
#y.retain_grad()
output = model(y)
act1 = activation['relu1']
act2 = activation['relu2']

act1 = act1.unsqueeze(0)
act2 = act2.unsqueeze(0)
imshow_grid(act1)
imshow_grid(act2)

