#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import PIL
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import matplotlib.pyplot as plt

##Importing Image
from PIL import Image
img = Image.open("baboon.jpg")

##Converting Image to Tensor
img_tensor = TF.to_tensor(img)
#print(img_tensor.shape)
n = 1                       #number of images (batches)
c = 3                       #number of channels (RGB)
w = img_tensor.size(dim=1)  #width
h =  img_tensor.size(dim=2) #height
img_tensor.resize_(n,c,w,h)
#print(img_tensor.shape)

ks = 10
#kernel_block = torch.einsum('i,jk->jk',ks2,kernel)
#print(kernel_block)
kernel_block = 1/ks**2 * torch.ones((ks,ks))
##Applying Convolution 1
weights = kernel_block
weights = weights.view(1, 1, kernel_block.size(dim=0), kernel_block.size(dim=1)).repeat(1, c, 1, 1)

output = F.conv2d(img_tensor, weights)
print(output.shape)
output = output.squeeze(0)
transform = T.ToPILImage()
img1 = transform(output)

##Applying Convolution 2

img_gray = TF.rgb_to_grayscale(img)
#img_gray.show()
img_gray_tensor = TF.to_tensor(img_gray)


ker = torch.tensor([[0., 0., 0.],
                    [1., 0., -1.],
                    [0., 0., 0.]])
kernel_block2 = 0.5*ker
weights2 = kernel_block2
weights2 = weights2.view(1, 1, kernel_block2.size(dim=0), kernel_block2.size(dim=1))
output2 = F.conv2d(img_gray_tensor, weights2)
output2 = output2.squeeze(0)
transform = T.ToPILImage()
img2 = transform(output2)

img_list = [img, img1, img2]
figure = plt.figure(figsize=(8,8))


plt.figure(1, figsize=(1, 3))
plt.clf()
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(img_list[i])
    plt.axis('off')

