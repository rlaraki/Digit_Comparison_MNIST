#!/usr/bin/env python3
"""
File: metrics.py
Description: utils file to store metrics functions 
"""
import torch
import math


def extract_mean_std(array):
    mean = sum(array) / len(array)
    std = math.sqrt(sum([pow(x - mean, 2) for x in array]) / len(array))

    return mean, std



# Compute accuracy for digit recognition and comparison.
def accuracy(model, data_loader, split, auxiliary=False, flatten=True):
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in data_loader:

            # Reshape input for linear model
            if flatten and (not split):
                inputs = inputs.view(-1, inputs.shape[1]*inputs.shape[2] * inputs.shape[3])
               
            # Compute model prediction
            if split | auxiliary:
                outputs_1, _, _ = model(inputs)
            else:
                outputs_1 = model(inputs)

            predicted = outputs_1.max(1)[1]

            # Compute accuracy
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    return acc
