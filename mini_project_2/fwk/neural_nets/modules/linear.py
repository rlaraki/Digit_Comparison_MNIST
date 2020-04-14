"""
File: linear.py
Description: Fully connected layer implementation
"""
from .module import Module
import torch
import math


class Linear(Module):

    def __init__(self, input_size, output_size, epsilon=1e-6):
        super(Linear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.epsilon = epsilon

        # Weights and biases Init
        self.weights = torch.empty(self.output_size, self.input_size).normal_(0, epsilon)
        self.bias = torch.empty(output_size).normal_(0, epsilon)

    def backward(self, grad_wr_to_output):
        return self.weights.t().mv(grad_wr_to_output)

    def forward(self, x):
        self.add_parameter('input', x)
        return self.weights.mv(x) + self.bias
