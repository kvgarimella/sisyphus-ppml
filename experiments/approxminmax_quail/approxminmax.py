import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ApproxMinMaxNorm2d(nn.Module):
    def __init__(self, num_features, a=2., b=1., mode=1):
        super(ApproxMinMaxNorm2d, self).__init__()
        self.register_buffer('running_max', torch.ones(1,num_features,1,1))
        self.register_buffer('running_min', torch.ones(1,num_features,1,1))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.register_buffer('mode', torch.tensor(mode, dtype=torch.long))# 0 -> biggest, 1 -> average, 2 -> real val
        self.a = a
        self.b = b
        
    def forward(self, input):
        if self.training:

            self.num_batches_tracked += 1
            if self.num_batches_tracked == 1:
                exponential_average_factor = 1.0 
            else:
                exponential_average_factor = 0.1

            ### Min block ###
            mins = input.amin(dim=(0,2,3), keepdim=True)

            if self.mode == 0:
                with torch.no_grad():
                    cond_min          = mins < self.running_min
                    self.running_min = torch.where(cond_min, mins, self.running_min) 

            elif self.mode == 1:
                with torch.no_grad():
                    self.running_min[:] = exponential_average_factor * mins + (1 - exponential_average_factor) * self.running_min

            input = input.sub(mins)
            ### Min block ###


            ### Max block ###
            maxs  = input.amax(dim=(0,2,3), keepdim=True) 
            
            if self.mode == 0:
                with torch.no_grad():
                    cond_max = maxs > self.running_max
                    self.running_max = torch.where(cond_max, maxs, self.running_max)

            elif self.mode == 1:
                with torch.no_grad():
                    self.running_max[:] = exponential_average_factor * maxs + (1 - exponential_average_factor) * self.running_max

            input = input.div(maxs)
            ### Max Block ###
        
        if not self.training:
            if self.mode != 2:
                input = input.sub(self.running_min)
                input = input.div(self.running_max)
            else:
                mins = input.amin(dim=(0,2,3), keepdim=True)
                input = input.sub(mins)
                maxs  = input.amax(dim=(0,2,3), keepdim=True) 
                input = input.div(maxs)

                
        
        return self.a * input - self.b 
class ApproxMinMaxNorm1d(nn.Module):
    def __init__(self, num_features, a=2., b=1., mode=1):
        super(ApproxMinMaxNorm1d, self).__init__()
        self.register_buffer('running_max', torch.ones(1,num_features))
        self.register_buffer('running_min', torch.ones(1,num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.register_buffer('mode', torch.tensor(mode, dtype=torch.long))# 0 -> biggest, 1 -> average, 2 -> real val
        self.a = a
        self.b = b

    def forward(self, input):
        if self.training:
      
            self.num_batches_tracked += 1
            if self.num_batches_tracked == 1:
                exponential_average_factor = 1.0 
            else:
                exponential_average_factor = 0.1

            ### Min block ###
            mins = input.amin(dim=(0), keepdim=True)

            if self.mode == 0:
                with torch.no_grad():
                    cond_min          = mins < self.running_min
                    self.running_min = torch.where(cond_min, mins, self.running_min) 

            elif self.mode == 1:
                with torch.no_grad():
                    self.running_min[:] = exponential_average_factor * mins + (1 - exponential_average_factor) * self.running_min

            input = input.sub(mins)
            ### Min block ###


            ### Max block ###
            maxs  = input.amax(dim=(0), keepdim=True) 
            
            if self.mode == 0:
                with torch.no_grad():
                    cond_max = maxs > self.running_max
                    self.running_max = torch.where(cond_max, maxs, self.running_max)

            elif self.mode == 1:
                with torch.no_grad():
                    self.running_max[:] = exponential_average_factor * maxs + (1 - exponential_average_factor) * self.running_max

            input = input.div(maxs)
            ### Max Block ###
        
        if not self.training:
            if self.mode != 2:
                input = input.sub(self.running_min)
                input = input.div(self.running_max)
            else:
                mins = input.amin(dim=(0), keepdim=True)
                input = input.sub(mins)
                maxs  = input.amax(dim=(0), keepdim=True) 
                input = input.div(maxs)

        return self.a * input - self.b 


def test1d():
    x = torch.randn(256,200)
    approx1d = ApproxMinMaxNorm1d(num_features=200, mode=2)
    x = approx1d(x)
    print("before")
    print(x.min(), x.max())
    print("after")
    approx1d.eval()
    x = approx1d(x)
    print("before")
    print(x.min(), x.max())
    print("after")


def test2d():
    x = torch.randn(256,3,32,32)
    print(x.amin(dim=(0,2,3), keepdim=True))
    print(x.amax(dim=(0,2,3), keepdim=True) - x.amin(dim=(0,2,3), keepdim=True))
    approx2d = ApproxMinMaxNorm2d(num_features=3, mode=2)
    print(approx2d.num_batches_tracked == 0)
    approx2d.train()
    x = approx2d(x)
    x = approx2d(x)
    print(x.min(), x.max())
    approx2d.eval()
    x = approx2d(x)
    print(x.min(), x.max())



#test1d()
#test2d()
