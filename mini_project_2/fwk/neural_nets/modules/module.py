"""
File: module.py
Description: Define the parent class for all neural network classes.
"""
from collections import OrderedDict


class Module(object):
    def __init__(self):
        self.__params = OrderedDict()
        self.weights = None
        self.bias = None

    def param(self):
        return self.__params

    def forward(self, *inputs):
        raise NotImplementedError

    def backward(self, *grad_wr_to_output):
        raise NotImplementedError

    def add_parameter(self, key, value):
        self.__params[key] = value

    def __call__(self, *args, **kwargs):
        return self.forward(*args)
