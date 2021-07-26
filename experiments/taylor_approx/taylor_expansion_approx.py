"""
Taylor Polynomial Approximations of
1. SiLU
2. ReLU

https://en.wikipedia.org/wiki/Taylor_series#Definition
"""

import torch
import torch.nn as nn

class SiLUTaylorApprox(nn.Module):
    def __init__(self, order=2):
        super(SiLUTaylorApprox, self).__init__()
        
        self.coeffs = torch.tensor([0.0, 1/2., 1/4., 0.0, -1/48, 0.0, 1/480, 0.0, -17/80640., 0.0, 31/1451520])
        self.order  = order
    
    def forward(self, x):
        out = 0
        for i in range(self.order+1):
            out = out + self.coeffs[i] * x**i
        return out

class ReLUTaylorApprox(nn.Module):
    def __init__(self, order=2):
        super(ReLUTaylorApprox, self).__init__()
        
        self.coeffs = torch.tensor([0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.order  = order
        
    def forward(self, x):
        out = 0
        for i in range(self.order+1):
            out = out + self.coeffs[i] * x**i
        return out

if __name__ == "__main__":
    s = SiLUTaylorApprox()
    r = ReLUTaylorApprox()
