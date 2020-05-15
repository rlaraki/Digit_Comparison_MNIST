#!/usr/bin/env python3

from collections import OrderedDict

import torch.nn as nn


class SiameseConv(nn.Module):   
    def __init__(self, d1, d2):
        super(SiameseConv, self).__init__()

        self.cnn_layers = nn.Sequential(
            OrderedDict(
                [
                    ("C1", nn.Conv2d(1, 6, kernel_size=2)),
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
                    ("F7", nn.Linear(84, 10)),
                    ("LogSoftmax", nn.LogSoftmax(dim=-1)),
                ]
            )
        )

    # Defining the forward pass    
    def forward_one(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.shape[0], -1)
        x = self.linear_layers(x)
        return out
    
    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        #  return self.sigmoid(out)
        return out
    
def simple_conv_net(**kwargs):
    """Wrapper for the neural network arguments"""
    model = SimpleConvNet(**kwargs)
    return model