#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch

class Cube(torch.autograd.Function):
    
    @staticmethod #tied to the class, not the object (you cannot access any data in the class)
    def forward(ctx,input):
        
        ctx.save_for_backward(input)
        return input**3
    
    @staticmethod
    def backward(ctx,grad_output):
        
        input, = ctx.saved_tensors #the comma unpacks the tuple
        #print('grad', input, grad_output) #grad_output is the u'*Dhf in the backward computation graph in the notes
                                        #you never need to calculate the jacobian!
        return 3*input**2 * grad_output
        pass
    
x = torch.arange(0,5,dtype=torch.float32,requires_grad=True) #input tensor
xc = torch.arange(0,5,dtype=torch.float32,requires_grad=True)

y = Cube.apply(x)
z = torch.mean(y) #take the mean of y


yc = xc**3 #correct calculation
zc = torch.mean(yc)

z.backward()
zc.backward()

#y.backward() #doesnt work because it expects a scalar output

np.testing.assert_almost_equal(y.detach().numpy(),yc.detach().numpy())
np.testing.assert_almost_equal(x.detach().numpy(),xc.detach().numpy())

# =============================================================================
# print(x.grad)
# print(xc.grad)
# =============================================================================
