from collections import OrderedDict

import torch
import torch.nn as nn

class Aux(nn.Module):
    def __init__(self, m1, ws, flatten):
        super(Aux, self).__init__()
    
        self.flatten = flatten
        self.m1 = m1
        self.ws = ws
        
        self.recognition_layer= nn.Sequential(
            OrderedDict(
                [
                    ("linear_1", nn.Linear(14*14, 75)),
                    ("Relu1", nn.ReLU()),
                    ("linear_2", nn.Linear(75, 50)),
                    ("Relu2", nn.ReLU()),
                    ("linear_3", nn.Linear(50, 10)),

                ]
            )
        )
        

    def forward(self, x):
        
        x1 = x[:,0,:,:]
        x2 = x[:,1,:,:]
        
       
        x1 = x1.view(-1, x1.shape[1] * x1.shape[2])
        x2 = x2.view(-1, x2.shape[1] * x2.shape[2])
    
            
        out1 = self.recognition_layer(x1)
        out2 = self.recognition_layer(x2)
        
        res = self.m1(x)
        
        return res, out1, out2


def aux(**kwargs):
    model = Aux(**kwargs)
    return model
