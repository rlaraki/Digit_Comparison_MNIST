#!/usr/bin/env python3
"""
File: LeNet.py
Author: Ambroise Renaud
Email: ambroise.renaud@gmail.com
Github: ambroisernd
Description: Pytorch implementation of LeNet (http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)
"""

from collections import OrderedDict

import torch
import torch.nn as nn

# TODO: Optimize this network
class LeNet(nn.Module):
    def __init__(self, d1, d2):
        super(LeNet, self).__init__()

        self.cnn_layers = nn.Sequential(
            OrderedDict(
                [
                    ("C1", nn.Conv2d(1, 6, kernel_size=(2, 2))),
                    ("Relu1", nn.ReLU()),
                    ("D1", nn.Dropout(d1)),
                    ("S2", nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
                    ("C3", nn.Conv2d(6, 7, kernel_size=(2, 2))),
                    ("Relu3", nn.ReLU()),
                    ("D2", nn.Dropout(d2)),
                    ("S4", nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
                    ("C5", nn.Conv2d(7, 120, kernel_size=(2,2))),
                    ("Relu5", nn.ReLU()),
                ]
            )
        )

        self.fully_connected = nn.Sequential(
            OrderedDict(
                [
                    ("F6", nn.Linear(120, 84)),
                    ("Relu6", nn.ReLU()),
                    ("F7", nn.Linear(84, 10)),
                    ("LogSoftmax", nn.LogSoftmax(dim=-1)),
                ]
            )
        )

    def forward(self, x):
        output = self.cnn_layers(x)
        output = output.view(x.shape[0], -1)
        output = self.fully_connected(output)
        return output


def lenet(**kwargs):
    model = LeNet(**kwargs)
    return model
