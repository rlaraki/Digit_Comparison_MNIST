"""
File: linear.py
Description: Fully connected layer implementation
"""
from .module import Module


class Linear(Module):

    def __init__(self):
        super(Linear, self).__init__()

    def backward(self, *grad_wr_to_output):
        pass

    def forward(self, *inputs):
        pass
