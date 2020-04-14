"""
File: relu.py
Description: ReLU activation implementation
"""
from .module import Module


class ReLU(Module):

    def __init__(self):
        super(ReLU, self).__init__()

    def backward(self, *grad_wr_to_output):
        pass

    def forward(self, *inputs):
        pass
