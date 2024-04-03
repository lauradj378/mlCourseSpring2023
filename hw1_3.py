#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import PIL
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import matplotlib.pyplot as plt


from sklearn.datasets import load_digits


digits = load_digits()
figure = plt.figure(figsize=(8,8))
cols, rows = 3, 3


plt.figure(1, figsize=(rows, cols))
plt.clf()
for i in range(9):
    plt.subplot(rows, cols, i+1)
    plt.gray()
    plt.imshow(digits.images[i])
    plt.axis('off')
    label = i
    plt.title(label)


F.one_hot(torch.arange(0, 10))

data = torch.utils.data.random_split(digits.images,[0.8, 0.2])
params = {
    'dim': 2,
    'width': 50,
    'depth': 3,
    'lr': 0.1,
    'epochs': 25,
    'batch_size': 10,
}
