"""
File: linear.py
Description: Fully connected layer implementation
"""
from .module import Module
import torch
import math


class Linear(Module):

    def __init__(self, input_size, output_size):
        self.input = None


        # Weights and biases Init
        self.weights = torch.empty(output_size, input_size).normal_()
        self.bias = torch.empty(output_size, 1).normal_()
        
        # Grad Init
        self.weights_grad = torch.zeros(self.weights.size())
        self.bias_grad = torch.zeros(self.bias.size())
      
    def forward(self, x):
        self.input = x
        return (self.weights.mm(x.t()) + self.bias).t()

    def backward(self, grad_wr_to_output):
        
        self.weights_grad += grad_wr_to_output.t().mm(self.input)
        self.bias_grad += grad_wr_to_output.t().sum(1).unsqueeze(1)
        
        return (self.weights.t().mm(grad_wr_to_output.t())).t()
    
    def param(self):
        return [(self.weights, self.weights_grad), 
                (self.bias, self.bias_grad)]
    
    def zero_grad(self):
        self.weights_grad.zero_()
        self.bias_grad.zero_()

