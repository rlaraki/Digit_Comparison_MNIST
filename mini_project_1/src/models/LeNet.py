#!/usr/bin/env python3
"""
Description: Pytorch implementation of LeNet (http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)
"""

from collections import OrderedDict

import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self, d=0, only_image=False):
        super(LeNet, self).__init__()

        input_shape = 1 if only_image else 2
        output_shape = 10 if only_image else 2

        self.cnn_layers = nn.Sequential(
            OrderedDict(
                [
                    ("C1", nn.Conv2d(input_shape, 6, kernel_size=(2, 2))),
                    ("Relu1", nn.ReLU()),
                    ("D1", nn.Dropout(d)),
                    ("S2", nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
                    ("C3", nn.Conv2d(6, 7, kernel_size=(2, 2))),
                    ("Relu3", nn.ReLU()),
                    ("D2", nn.Dropout(d)),
                    ("S4", nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
                    ("C5", nn.Conv2d(7, 120, kernel_size=(2, 2))),
                    ("Relu5", nn.ReLU()),
                ]
            )
        )

        self.fully_connected = nn.Sequential(
            OrderedDict(
                [
                    ("F6", nn.Linear(120, 84)),
                    ("Relu6", nn.ReLU()),
                    ("F7", nn.Linear(84, output_shape)),
                    ("LogSoftmax", nn.LogSoftmax(dim=-1)),
                ]
            )
        )

    def forward(self, x):
        output = self.cnn_layers(x)
        output = output.view(x.shape[0], -1)
        output = self.fully_connected(output)
        return output


# Build network for the full task (image recognition and comparison)
def le_net(dropout=0, only_image=False):
    model = LeNet(dropout, only_image)
    return model




