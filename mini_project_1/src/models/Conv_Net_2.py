
from collections import OrderedDict

import torch.nn as nn


class Conv_Net_2(nn.Module):   
    def __init__(self, d1, d2):
        super(Conv_Net_2, self).__init__()
        
        
        self.cnn_layers = nn.Sequential(
            OrderedDict(
                [
                    ("C1", nn.Conv2d(2, 6, kernel_size=2)),
                    ("BN1", nn.BatchNorm2d(6)),
                    ("Relu1", nn.ReLU()),
                    ("D1", nn.Dropout(d1)),
                    ("S2", nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
                    ("C3", nn.Conv2d(6, 7, kernel_size=2)),
                    ("BN2", nn.BatchNorm2d(7)),
                    ("Relu3", nn.ReLU()),
                    ("D2", nn.Dropout(d2)),
                    ("S4", nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
                    ("C5", nn.Conv2d(7, 100, kernel_size=2)),
                    ("BN3", nn.BatchNorm2d(100)),
                    ("Relu5", nn.ReLU()),
                ]
            )
        )

        self.linear_layers = nn.Sequential(
            OrderedDict(
                [
                    ("F6", nn.Linear(100, 84)),
                    ("Relu6", nn.ReLU()),
                    ("F7", nn.Linear(84, 2)),
                    ("LogSoftmax", nn.LogSoftmax(dim=-1)),
                ]
            )
        )

    # Defining the forward pass    
    def forward(self, x):

        out = self.cnn_layers(x)
        out = out.view(x.shape[0], -1)
        res = self.linear_layers(out)
        
        return res
    
def conv_net_2(**kwargs):
    """Wrapper for the neural network arguments"""
    model = Conv_Net_2(**kwargs)
    return model