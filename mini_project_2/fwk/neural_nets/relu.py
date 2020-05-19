"""
File: relu.py
Description: ReLU activation implementation
"""
from .module import Module
import torch

class ReLU(Module):

    def __init__(self):
        self.input = None

    def __sigma(self, x):
        return torch.max(x, torch.zeros_like(x))

    def __dsigma(self, x):
        return torch.where(x <= 0, torch.tensor(0.), torch.tensor(1.))

    def backward(self, grad_wr_to_output):
        return self.__dsigma(self.input) * grad_wr_to_output

    def forward(self, x):
        self.input = x
        return self.__sigma(x)
    
    def zero_grad(self):
        pass

