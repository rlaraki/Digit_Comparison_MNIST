"""
File: mse.py
Description: MSE loss function implementation
"""
from .module import Module


class MSE(Module):
    
    def __init(self):
        self.error = None


    def __loss(self, error):
        return error.pow(2).sum()

    def __dloss(self, error):
        return 2 * error
    
    
    def forward(self, input_, target):
        self.error = input_ - target
        return self.__loss(self.error)

    def backward(self):
        return self.__dloss(self.error)
       


