"""
File: linear.py
Description: Fully connected layer implementation
"""
from .module import Module
import torch
import math


class Linear(Module):

    def __init__(self, input_size, output_size):
        super(Linear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        # Weights and biases Init
        self.weights = torch.empty((self.output_size, self.input_size))
        stdv = 1. / math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-stdv, stdv)
        self.bias = torch.empty(output_size)  # TODO: check if zero allowed
        self.bias.data.uniform_(-stdv, stdv)

    def backward(self, grad_wr_to_output):
        return self.weights.t().mv(grad_wr_to_output)

    def forward(self, x):
        self.add_parameter('input', x)
        return self.weights.mv(x) + self.bias
