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
from torchvision.transforms import Lambda
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import numpy as np

params = {
    'network': LeNet5,
    'learning_rate': 1e-3,
    'batch_size': 64,
    'epochs': 40,
    'n_classes': 10}

network = params['network']()
network.load_state_dict(torch.load('/Users/laurahull/Documents/UCF/Spring_23/Machine_Learning_Course/dl/LeNet-5-MNIST.pth'))
network.eval()

kernels1 = network.conv1.weight.detach().clone()
kernels2 = network.conv2.weight.detach().clone()

def imshow_grid(img):
    '''
    Input "img" must be converted to a tensor before being called in this function
    '''
    m = img.size(dim=0)
    n = img.size(dim=1)
    height = img.size(dim=2)
    width = img.size(dim=3)
    img = F.pad(img, (1,1,1,1), mode='constant', value=np.nan)
    m = img.size(dim=0)
    n = img.size(dim=1)
    height = img.size(dim=2)
    width = img.size(dim=3)
    img = img.permute(0,2,1,3)
    img = torch.reshape(img,([m*height, n*width]))
    plt.figure()
    plt.imshow(img)
    plt.axis('off')
imshow_grid(kernels1)
imshow_grid(kernels2)


activation = {}
def get_activation(name):
    def hook(network, input, output):
        activation[name] = output.detach()
    return hook

model = LeNet5()
model.eval()
model.relu1.register_forward_hook(get_activation('relu1'))
model.relu2.register_forward_hook(get_activation('relu2'))

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size = params['batch_size'],shuffle = False)
image, _ = training_data[0]

y = image #already pre-processed
#print(y.size())
y.requires_grad_()
output = model(y)
max_output = output.max()
max_output.backward()

act1 = activation['relu1']
act2 = activation['relu2']
#print(type(act1))

act1 = act1.unsqueeze(0)
act2 = act2.unsqueeze(0)
#print(type(act1))
imshow_grid(act1)
imshow_grid(act2)

saliency_map = (torch.abs(image.grad) * max_output).squeeze().detach().numpy()

print(image.squeeze().size())
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image.squeeze().detach().numpy(), cmap='gray')
ax[0].axis('off')
ax[0].set_title(f'Input Image')
ax[1].imshow(saliency_map, cmap='jet')
ax[1].axis('off')
ax[1].set_title('Saliency Map')


'''
HW 3.3: Visualize AlexNet
'''

alexnet = models.alexnet(pretrained=True)
alexnet.eval()
##Importing Image
from PIL import Image
img = Image.open("n01440764_tench.jpeg")

##Converting Image to Tensor
img_tensor = TF.to_tensor(img)
img_tensor.requires_grad_()
print(img_tensor)

resized_img = TF.resize(img_tensor,size=(224,224))
resized_img = resized_img.unsqueeze(0)
resized_img = resized_img.detach()
resized_img.requires_grad = True
#print(resized_img)
#resized_img.requires_grad()
# print(resized_img.size())
# print(type(resized_img))
# output = alexnet(resized_img)
print(alexnet)

conv_layers = alexnet.features

for i, layer in enumerate(conv_layers):
    if isinstance(layer, torch.nn.Conv2d):
        # Access the convolutional kernel weights
        kernels = layer.weight.detach().clone()
        imshow_grid(kernels)

relu_outputs = {}

def hook_fn(module, input, output):
    relu_outputs['relu'] = output.detach()

# alexnet.features[4].register_forward_hook(hook_fn)
output = alexnet(resized_img)
max_output = output.max()
max_output.backward()
# relu_outputs = relu_outputs['relu']

# imshow_grid(relu_outputs)
#print(alexnet.features[1])
#print(type(conv1_relu_output))
#relu_outputs = relu_outputs['relu']
#imshow_grid(relu_outputs)

conv1_relu_output = alexnet.features[0](resized_img)
relu1_output = nn.ReLU(inplace=True)(conv1_relu_output)
conv2_relu_output = alexnet.features[3](conv1_relu_output)
relu2_output = nn.ReLU(inplace=True)(conv2_relu_output)
conv3_relu_output = alexnet.features[6](conv2_relu_output)
relu3_output = nn.ReLU(inplace=True)(conv3_relu_output)
conv4_relu_output = alexnet.features[8](conv3_relu_output)
relu4_output = nn.ReLU(inplace=True)(conv4_relu_output)
conv5_relu_output = alexnet.features[10](conv4_relu_output)
relu5_output = nn.ReLU(inplace=True)(conv5_relu_output)

imshow_grid(relu1_output.detach())
imshow_grid(relu2_output.detach())
imshow_grid(relu3_output.detach())
imshow_grid(relu4_output.detach())
imshow_grid(relu5_output.detach())

#print(resized_img.grad)

saliency_map = (torch.abs(resized_img.grad) * max_output).squeeze().detach().numpy()

print(resized_img.size())
print(type(resized_img))
print(resized_img.squeeze().size())
resized_img = resized_img.squeeze().detach().numpy()
fig, ax = plt.subplots(1, 2)
ax[0].imshow(resized_img.transpose(1, 2, 0), cmap='gray')
ax[0].axis('off')
ax[0].set_title(f'Input Image')
ax[1].imshow(saliency_map.transpose(1, 2, 0), cmap='jet')
ax[1].axis('off')
ax[1].set_title('Saliency Map')


plt.show()