"""
File: mse.py
Description: MSE loss function implementation
"""
from .module import Module
import torch


class MSE(Module):

    def __init__(self):
        super(MSE, self).__init__()
        self.__loss_value = None

    def __loss(self, input, target):
        return ((input - target) ** 2).mean()

    def __dloss(self, input, target):
        return 2 * (- (target - input))

    def backward(self, model):
        grad = self.__dloss(self.param().get('input'), self.param().get('target'))
        model.backward(grad)
        return grad

    def forward(self, input, target):
        self.add_parameter('input', input)
        self.add_parameter('target', target)
        self.__loss_value = self.__loss(input, target)
        return self

    def item(self):
        return self.__loss_value.item()
