"""
File: optimizer.py
Description: Define the parent class for all optimizer classes.
"""


class Optimizer(object):

    def step(self):
        raise NotImplementedError

