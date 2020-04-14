"""
File: optimizer.py
Description: Define the parent class for all optimizer classes.
"""


class Optimizer(object):
    def __init__(self, params):
        self.__params = params

    def step(self):
        raise NotImplementedError

    def param(self):
        return self.__params
