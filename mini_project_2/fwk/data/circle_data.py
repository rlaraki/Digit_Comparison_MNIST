"""
File: circle_data.py
Description: Project 2 circle dataset (cf project 2 sheet)
"""
import torch
import math


def generate_data(size=1000):
    # One hot
    zero = torch.tensor([0.])
    one = torch.tensor([1.])
    # Random
    x = torch.rand(size)
    y = torch.rand(size)

    # Create pairs
    dataset = [torch.tensor([x_i, y_i]) for x_i, y_i in zip(x, y)]
    # Compute distance from radius
    distances = (torch.pow(x - 0.5, 2) + torch.pow(y - 0.5, 2)).sqrt()
    labels = torch.where(distances > (1 / math.sqrt((2 * math.pi))), zero, one)
    # Populate one-hot labels
    one_hot = []
    for x in labels:
        if x.item() > .5:
            one_hot.append(torch.tensor([0., 1.]))
        else:
            one_hot.append(torch.tensor([1., 0.]))
    return dataset, one_hot
