"""
File: relu.py
Description: Tanh activation implementation
"""
from .module import Module


class Tanh(Module):

    def __init__(self):
        super(Tanh, self).__init__()

    def backward(self, *grad_wr_to_output):
        pass

    def forward(self, *inputs):
        pass