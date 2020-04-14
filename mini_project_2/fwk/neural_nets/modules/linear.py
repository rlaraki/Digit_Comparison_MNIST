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
        self.weights = torch.zeros((self.output_size, self.input_size))
        self.bias = torch.zeros(output_size)  # TODO: check if zero allowed

    def backward(self, grad_wr_to_output):
        return self.weights.t().mv(grad_wr_to_output)

    def forward(self, x):
        self.add_parameter('input', x)
        return self.weights.mv(x) + self.bias
