"""
File: mse.py
Description: MSE loss function implementation
"""
from .module import Module


class MSE(Module):

    def __init__(self):
        super(MSE, self).__init__()
        self._loss_value = None

    def backward(self, *grad_wr_to_output):
        pass

    def forward(self, *inputs):
        self._loss_value = 42  # Dummy value
        return self

    def item(self):
        return self._loss_value.item()
