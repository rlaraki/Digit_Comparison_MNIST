#!/usr/bin/env python3
"""
File: SimpleConvNet.py
Author: Pierre Schutz
Github: pierreschutz
Description: Pytorch implementation of a basic convolutional neural network
"""

from collections import OrderedDict

import torch.nn as nn


class SimpleConvNet(nn.Module):   
    def __init__(self, d1, d2):
        super(SimpleConvNet, self).__init__()

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
    def forward(self, x):
        out = self.cnn_layers(x)
        out = out.view(x.shape[0], -1)
        out = self.linear_layers(out)
        return out
    
def simple_conv_net(**kwargs):
    """Wrapper for the neural network arguments"""
    model = SimpleConvNet(**kwargs)
    return model

#creer methode comme ça simple_conv_net_drop0.2 / 0.7 ...