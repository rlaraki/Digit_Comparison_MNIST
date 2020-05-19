"""
File: relu.py
Description: Tanh activation implementation
"""
from .module import Module


class Tanh(Module):

    def __init__(self):
        self.input = None

    def __sigma(self, x):
        return x.tanh()

    def __dsigma(self, x):
        return 1 - x.tanh().pow(2)

    def backward(self, grad_wr_to_output):
        return self.__dsigma(self.input) * grad_wr_to_output

    def forward(self, x):
        self.input = x
        return self.__sigma(x)
    
    def zero_grad(self):
        pass