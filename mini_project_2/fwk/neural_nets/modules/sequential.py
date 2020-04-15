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
        derivative = []  # Queue to store last compute s(l)/dx
        for key in reversed_layers:
            layer = self.layers.get(key)
            prev_d = layer.backward(prev_d)
            if layer.weights is not None:

                # Add grad
                if self.grads.get(key) is not None:
                    if layer.bias is None:
                        self.grads[key][0].add_(derivative[-1].view(-1, 1).mm(layer.param().get('input').view(1, -1)))
                        derivative.pop(-1)
                    else:
                        self.grads[key][0].add_(derivative[-1].view(-1, 1).mm(layer.param().get('input').view(1, -1)))
                        self.grads[key][1].add_(derivative[-1])
                    derivative.pop(-1)
                # Init key
                elif self.grads.get(key) is None:
                    if layer.bias is None:
                        self.grads[key] = (derivative[-1].view(-1, 1).mm(layer.param().get('input').view(1, -1)), None)
                        derivative.pop(-1)
                    else:
                        self.grads[key] = (
                            derivative[-1].view(-1, 1).mm(layer.param().get('input').view(1, -1)), derivative[-1])
                        derivative.pop(-1)
            else:
                derivative.append(prev_d)
        return self.grads
