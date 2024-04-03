#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implement Newton's method
$$
  x^{n+1} = x^n - \\frac{f(x^n)}{f'(x^n)},
$$
as outlined in the given *Python file*. For the derivative, use the autograd 
mechanism in Pytorch.
The included test, checks your implementation for finding the zero of
$$\\begin{aligned}
  f(x) & = 0, &
  f(x) & = \\sin(x) - \\cos(x)
\\end{aligned}$$
with initial value $x^0 = 0$ and $7$ steps.
"""
import unittest
import numpy as np
import torch

x0 = 0.
n_steps = 7.

def newton(f, x0: float, n_steps: int) -> float:
    """
    Implementation of Newton's method.
    The inputs and outputs are Python floats, not torch.Tensor.
    Args:
        f: Function, should work with Python floats and torch.Tensor.
        x0: Initial value for Newton's method
        n_steps: Number of Newton steps.
    Returns:
        The output of Newton's method.
    """
    for i in range(0,int(n_steps)):
        x0 = torch.as_tensor(x0,dtype=torch.float64)
        x0 = x0.clone().detach().requires_grad_(True)
        func = f(x0)
        func.backward()
        x0 = x0 - func/x0.grad
    return float(x0)

def f(x):
    return torch.sin(x) - torch.cos(x)
class TestNewtonsMethod(unittest.TestCase):
    def test_value(self):
        y = newton(f, x0=0., n_steps=7)
        np.testing.assert_almost_equal(y, np.pi/4)
if __name__ == '__main__':
    unittest.main()
    
    """
    OUTPUTS:
       runfile('/Users/laurahull/Documents/UCF/Spring 23/Machine Learning Course/dl/assignments/EXAM_P3.py', wdir='/Users/laurahull/Documents/UCF/Spring 23/Machine Learning Course/dl/assignments')
       .
       ----------------------------------------------------------------------
       Ran 1 test in 0.002s

       OK
       """
