"""
File: module.py
Description: Define the parent class for all neural network classes.
"""
from collections import OrderedDict


class Module(object):

    def forward(self, *inputs):
        raise NotImplementedError

    def backward(self, *grad_wr_to_output):
        raise NotImplementedError
    
    def param(self):
        return []
    
    def zero_grad(self):
        pass
    

