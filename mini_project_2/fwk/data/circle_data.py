"""
File: circle_data.py
Description: Project 2 circle dataset (cf project 2 sheet)
"""
import torch
import math


def generate_data(size=1000):
    # One hot
    dataset = torch.Tensor(1000, 2).uniform_(0, 1)
    
    # Compute distance from radius
    distances = torch.pow(dataset - 0.5, 2).sum(1).sqrt()
    labels = torch.where(distances > (1 / math.sqrt((2 * math.pi))), torch.Tensor([0]), torch.Tensor([1]))
    one_hot = torch.Tensor([[0., 1.] if l.item() == 1. else [1., 0.] for l in labels])
    
    return dataset, labels, one_hot
