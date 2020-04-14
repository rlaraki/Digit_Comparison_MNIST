"""
File: sequential.py
Description: Sequential module to compute back propagation
"""
from .module import Module
from collections import OrderedDict


class Sequential(Module):

    def __init__(self, dict):
        super(Sequential, self).__init__()
        self.layers = dict
        self.grads = OrderedDict()

    def forward(self, x):
        crt_out = x
        for key in self.layers:
            crt_out = self.layers.get(key)(crt_out)
        return crt_out

    def backward(self, d_loss):
        reversed_layers = OrderedDict(reversed(list(self.layers.items())))
        prev_d = d_loss
        for key in reversed_layers:
            layer = self.layers.get(key)
            prev_d = layer.backward(prev_d)
            if layer.weights is not None:
                if layer.bias is None:
                    if self.grads[key] is None:
                        self.grads[key] = (prev_d.view(-1, 1).mm(layer.param().get('input').view(1, -1)), None)
                else:
                    self.grads[key] = (prev_d.view(-1, 1).mm(layer.param().get('input').view(1, -1)), prev_d)
        return self.grads
