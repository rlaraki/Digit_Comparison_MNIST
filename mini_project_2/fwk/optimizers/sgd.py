"""
File: sgd.py
Description: SGD implementation.
"""
from .optimizer import Optimizer


class SGD(Optimizer):

    def __init__(self, params):
        super(SGD, self).__init__(params)

    def step(self):
        print(self.param().get('grads'))
