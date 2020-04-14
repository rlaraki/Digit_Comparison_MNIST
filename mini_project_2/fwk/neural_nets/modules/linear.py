"""
File: linear.py
Description: Fully connected layer implementation
"""
from .module import Module
import torch


class Linear(Module):

    def __init__(self, input_size, output_size):
        super(Linear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weights = torch.empty((self.input_size, self.output_size))
        self.bias = torch.empty(output_size)

    def backward(self, *grad_wr_to_output):
        pass

    def forward(self, x):
        return self.weights.mv(x)