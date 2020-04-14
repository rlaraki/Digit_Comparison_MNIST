"""
File: module.py
Description: Define the parent class for all neural network classes.
"""
from collections import OrderedDict


class Module(object):
    def __init__(self):
        self._params = OrderedDict()

    def param(self):
        return self._params

    def forward(self, *inputs):
        raise NotImplementedError

    def backward(self, *grad_wr_to_output):
        raise NotImplementedError

    def add_parameter(self, key, value):
        self._params[key] = value

    def __call__(self, *args, **kwargs):
        return self.forward(*args)
