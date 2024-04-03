#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: laurahull
"""

import torch
import torch.nn as nn

#batch = number of images
#channels = RGB
#height = pixel height
#width = pixel width

class ConveDemonstration(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels = 3, out_channels = 1, kernel_size = (3,3))
        self.linear = nn.Linear(in_features = 324, out_features = 5)
        
    def forward(self,x):
        y = self.conv(x)
        y = torch.flatten(y,start_dim=1) #flattens indices 1,2,3 (leaves 0 alone)
        y = self.linear(y)
        return y
    
net = ConveDemonstration()

batch, channels, height, width = [10,3,20,20]
x = torch.zeros((batch,channels,height,width))
print(net(x).shape) #output images should be (18,18) because we haven't applied any padding
