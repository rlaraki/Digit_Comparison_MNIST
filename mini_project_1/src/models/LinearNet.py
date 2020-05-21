#!/usr/bin/env python3
"""
Description: Pytorch implementation of a neural network with linear layer and relu activation funtions
"""

from collections import OrderedDict

import torch.nn as nn


class LinearNet(nn.Module):
    """Pytorch implementation of a neural network with linear layer and relu activation funtions"""

    def __init__(self, only_image=False, input_size=14 * 14, num_classes=10):
        super(LinearNet, self).__init__()
        
        self.only_image = only_image

        self.input_size = input_size if only_image else 2 * input_size
        self.num_classes = num_classes if only_image else 2

        self.net = nn.Sequential(
            OrderedDict(
                [
                    ("linear_1", nn.Linear(self.input_size, 75)),
                    ("Relu1", nn.ReLU()),
                    ("linear_2", nn.Linear(75, 50)),
                    ("Relu2", nn.ReLU()),
                    ("linear_3", nn.Linear(50, self.num_classes)),

                ]
            )
        )

    def forward(self, x):
        if not self.only_image:
            x = x.view(-1, x.shape[1], x.shape[2] * x.shape[3])
                
        out = self.net(x)
        return out


def linear_net(only_image=False):
    model = LinearNet(only_image)
    return model
