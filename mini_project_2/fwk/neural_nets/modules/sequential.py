"""
File: sequential.py
Description: Sequential module to compute back propagation
"""
from .module import Module


class Sequential(Module):

    def __init__(self, dict):
        super(Sequential, self).__init__()
        self.layers = dict

    def forward(self, x):
        crt_out = x
        for key in self.layers:
            crt_out = self.layers.get(key)(crt_out)
        return crt_out

    def backward(self, *grad_wr_to_output):
        pass
