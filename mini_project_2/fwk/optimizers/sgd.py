"""
File: sgd.py
Description: SGD implementation.
"""
from .optimizer import Optimizer


class SGD(Optimizer):

    def __init__(self, params, eta=0.1):
        super(SGD, self).__init__(params)
        self.eta = eta

    def step(self):
        grads = self.param().get('grads')
        layers = self.param().get('layers')
        for key in grads:
            layer = layers.get(key)
            grad = grads.get(key)
            if grad[1] is not None: # Check if bias
                layer.weights -= self.eta * grad[0]
                layer.bias -= self.eta * grad[1]
            else:
                layer.weights -= self.eta * grad[0]