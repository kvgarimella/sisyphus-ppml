"""
Polynomial Regression

1. SiLU
2. ReLU

https://en.wikipedia.org/wiki/Polynomial_regression#Definition_and_example
"""

import torch
import torch.nn as nn
from generate_poly_regression_coeffs import *

class SiLUPolyApprox(nn.Module):
    def __init__(self, R=10, granularity=1/1000, order=2):
        super(SiLUPolyApprox, self).__init__()
        
        self.coeffs = torch.tensor([generate_coeffs(silu, R, granularity, order)]).squeeze()
        self.order  = order
    
    def forward(self, x):
        out = 0
        for i in range(self.order+1):
            out = out + self.coeffs[i] * x**i
        return out

class ReLUPolyApprox(nn.Module):
    def __init__(self, R=10, granularity=1/1000, order=2):
        super(ReLUPolyApprox, self).__init__()
        
        self.coeffs = torch.tensor([generate_coeffs(relu, R, granularity, order)]).squeeze()
        self.order  = order
        
    def forward(self, x):
        out = 0
        for i in range(self.order+1):
            out = out + self.coeffs[i] * x**i
        return out

if __name__ == "__main__":
    s = SiLUPolyApprox()
    r = ReLUPolyApprox()
