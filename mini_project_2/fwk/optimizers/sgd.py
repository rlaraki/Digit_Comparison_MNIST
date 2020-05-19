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
        for (param, grad) in self.params: # Iterate of the modules of the neural network
            param.sub_(grad*self.eta) # Update the weights

    def set_eta(self, value):
        self.eta = value
