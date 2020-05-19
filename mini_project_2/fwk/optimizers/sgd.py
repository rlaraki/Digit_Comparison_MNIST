"""
File: sgd.py
Description: SGD implementation.
"""
from .optimizer import Optimizer


class SGD(Optimizer):

    def __init__(self, params, eta=1e-3):
        self.params = params
        self.eta = eta

    def step(self):
        for layer_param in self.params:
            for (weight, grad) in layer_param:
                weight.sub_(grad*self.eta)

    def set_eta(self, value):
        self.eta = value
