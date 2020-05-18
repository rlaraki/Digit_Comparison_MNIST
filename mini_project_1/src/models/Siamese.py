from collections import OrderedDict

import torch.nn as nn
import torch

class Siamese(nn.Module):   
    def __init__(self, m1, ws):
        super(Siamese, self).__init__()
        self.ws = ws
        self.m1 = m1               
        self.m2 = m1
        
        
        self.linear_ws = nn.Sequential(
                OrderedDict(
                    [("F6", nn.Linear(20, 100)),
                    ("Relu6", nn.ReLU()),
                    ("F7", nn.Linear(100, 2)),
                    ("LogSoftmax", nn.LogSoftmax(dim=-1))]))
                    
    

    # Defining the forward pass    
    
    def forward(self, x):
        
        out1 = self.m1(x[:,0,:,:].unsqueeze(1))
        if self.ws:
            out2 = self.m1(x[:,1,:,:].unsqueeze(1))
        else:
            out2 = self.m2(x[:,1,:,:].unsqueeze(1))
        tot = torch.cat((out1, out2), 1)  #10*20
        res = self.linear_ws(tot)
        return res, out1, out2
    
def siamese(**kwargs):
    """Wrapper for the neural network arguments"""
    model = Siamese(**kwargs)
    return model
