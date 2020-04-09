#!/usr/bin/env python3
"""
File: LinearReluNet.py
Author: Ambroise Renaud
Email: ambroise.renaud@gmail.com
Github: ambroisernd
Description: Pytorch implementation of a neural network with linear layer and relu activation funtions
"""

from collections import OrderedDict

import torch.nn as nn


class LinearReluNet(nn.Module):

    """Pytorch implementation of a neural network with linear layer and relu activation funtions"""

    def __init__(self, input_size, nb_classes):
        super(LinearReluNet, self).__init__()

        self.input_size = input_size
        self.nb_classes = nb_classes

        self.net = nn.Sequential(
            OrderedDict(
                [
                    ("linear_1", nn.Linear(input_size, 75)),
                    ("Relu1", nn.ReLU()),
                    ("linear_2", nn.Linear(75, 50)),
                    ("Relu2", nn.ReLU()),
                    ("linear_3", nn.Linear(50, nb_classes)),
                ]
            )
        )

    def forward(self, x):
        out = self.net(x)
        return out


def linear_relu_net(**kwargs):
    """Wrapper for the neural network arguments"""
    model = LinearReluNet(**kwargs)
    return model
