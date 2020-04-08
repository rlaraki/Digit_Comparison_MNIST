#!/usr/bin/env python3
"""
File: metrics.py
Description: utils file to store metrics functions 
"""
import torch
def accuracy(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in data_loader:
            inputs_1 = inputs[:, 0, :, :].to(device)
            inputs_2 = inputs[:, 1, :, :].to(device)
            
            inputs_1 = inputs_1.view(-1, inputs_1.shape[1] * inputs_1.shape[2])
            inputs_2 = inputs_2.view(-1, inputs_2.shape[1] * inputs_2.shape[2])
            
            outputs_1 = model(inputs_1)
            outputs_2 = model(inputs_2)

            _, predicted_1 = outputs_1.max(1)
            _, predicted_2 = outputs_2.max(1)
            predicted = predicted_1 <= predicted_2
            correct += (predicted.cpu() == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    return acc

