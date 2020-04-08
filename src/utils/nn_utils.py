#!/usr/bin/env python3
"""
File: nn_utils.py
Description: neural network utils 
"""
import torch


def train_model(model, num_epochs, train_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Copy all model parameters to the GPU
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        total_loss = 0.0

        for (inputs, labels) in train_loader:
            for i in range(2):  # train on each element of the pair
                inputs_single = inputs[:, i, :, :].to(device)
                labels_single = labels.t()[i].to(device)
                inputs_single = inputs_single.view(
                    -1, inputs_single.shape[1] * inputs_single.shape[2]
                )
                optimizer.zero_grad()
                outputs = model(inputs_single)

                loss = criterion(outputs, labels_single.long())
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            print("Epoch %d, Loss=%.4f" % (epoch + 1, total_loss / len(train_loader)))
