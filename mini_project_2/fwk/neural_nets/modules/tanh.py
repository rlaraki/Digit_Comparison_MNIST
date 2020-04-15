"""
File: relu.py
Description: Tanh activation implementation
"""
from .module import Module


class Tanh(Module):

    def __init__(self):
        super(Tanh, self).__init__()

    def __sigma(self, x):
        return x.tanh()

    def __dsigma(self, x):
        return 4 * (x.exp() + x.mul(-1).exp()).pow(-2)

    def backward(self, grad_wr_to_output):
        return self.__dsigma(self.param().get('input')) * grad_wr_to_output

    def forward(self, x):
        self.add_parameter('input', x)
        return self.__sigma(x)