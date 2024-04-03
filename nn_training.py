#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 14:30:00 2023

@author: laurahull
"""

#hidden layer h(x) = wx + b
#f(x) = A Relu (wx + b)

import itertools
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

#Parameters
width = 50
learning_rate = 0.1
n_iter = 10
torch.manual_seed(0)
batch_size = 10
n_samples = 10
depth = 1
in_dim = 1
out_dim = 1
#Data
x = torch.linspace(-1,1,n_samples).reshape([-1,1])
y = x**2

dataset = list(zip(x,y))
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True) #splits data into batches
print(dataset)

#Network

#class NetBase: 
#    def __init__(self):
#        
#        self._modules = []
#        
#    def __setattr__(self,name,value):
#        object.__setattr__(self,name,value)
#             
#        if isinstance(value,nn.Module):
#                 
#            self._modules.append(value)
#    
#    def parameters(self):
#        
#        return itertools.chain(*(m.parameters() for m in self._modules))
#        #return itertools.chain(self.linear1.parameters(), self.linear2.parameters())
#        #return [self.b, self.W, self.A]
#    def __call__(self,*args,**kwargs):
#        
#       return self.forward(*args,**kwargs)

class Net(nn.Module):
        
    
   def __init__(self,width):
       
       super().__init__()
    
       #self.W = torch.randn((width,1),requires_grad=True)
       #self.b = torch.empty(width).uniform_(-1,1) #randomly sampled w/ uniform distribution
       #self.b.requires_grad_()
       #self.A = torch.randn((1,width),requires_grad=True)
       
       self.layers = nn.Sequential(
           nn.Linear(in_features=1,out_features=width,bias=True),
           nn.ReLU(),
           #nn.Linear(in_features=width,out_features=width,bias=True),
           #nn.ReLU(),
           nn.Linear(in_features=width,out_features=1,bias=False),
       )
       #print(type(self.layers))
       #self.linear1 = nn.Linear(in_features=1,out_features=width,bias=True)
       #self.relu = nn.ReLU()
       #self.linear2 = nn.Linear(in_features=width,out_features=1,bias=False)
       
      # print('Module?',isinstance(self.linear1,nn.Module)) #asking if linear1 is receiving the parameters

   def forward(self,x):
       
       return self.layers(x)
        

#Train

def train(net, dataloader,  optimizer, n_iter):
    
    for epoch in range(n_iter):
        print('epoch', epoch)
        
        for i, (x,y) in enumerate(dataloader):

            pred = net(x)
            loss = F.mse_loss(pred,y)
        
            optimizer.zero_grad()
            loss.backward() #computes gradient descent for all  of the variables 
                    #which we have told Python we need a gradient (W,b,A)
            optimizer.step()
        #for p in net.parameters():
         #   p.data -= learning_rate * p.grad #performs gradient  descent for each parameter in parameters function
        
            print(f'iter{i}, loss{loss}')
        
    return pred, loss

if __name__ == '__main__': #this doesn't run if you want to import the above functions to other files
   
    net = Net(width) #instantiating the class Net (similar to calling a function)
    #net = Net(in_dim, out_dim, width, depth)
    optimizer = torch.optim.SGD(net.parameters(),lr = learning_rate)
    #print(net._modules)
    pred, loss = train(net, dataloader, optimizer, n_iter)

    #print('net is instance of Net:', isinstance(net,Net))
    #print('net is instance of NetBase:', isinstance(net,NetBase))

#Plot
    plt.plot(x,y,color = 'red')
    plt.plot(x,pred.detach().numpy(),color = 'blue')
    plt.show()

#Poor excuse of a unit test!!!
    #np.testing.assert_almost_equal(loss.detach().numpy(),0.00015555770369246602) #sanity check to make sure
                            #I don't break anything when changing the code