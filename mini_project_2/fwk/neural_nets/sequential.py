"""
File: sequential.py
Description: Sequential module to compute back propagation
"""
from .module import Module
from collections import OrderedDict


class Sequential(Module):

    def __init__(self, dict_):
        self.layers = dict_

    def forward(self, x):
        crt_out = x
        for key in self.layers:
            crt_out = self.layers.get(key).forward(crt_out)
        return crt_out

    def backward(self, d_loss):
        
        reversed_layers = OrderedDict(reversed(list(self.layers.items())))
        prev_d = d_loss
        
        for key in reversed_layers:
            prev_d = self.layers.get(key).backward(prev_d)
        return prev_d
    
    def param(self):
        return [p for key in self.layers for p in self.layers.get(key).param()]
     
    def zero_grad(self):
        for key in self.layers:
            self.layers.get(key).zero_grad()
