#!/usr/bin/env python3
"""
File: main.py
Description: Mini project 1 main file
"""
import torch
from torch.utils import data

from models.LeNet import lenet
from models.LinearReluNet import linear_relu_net
from utils.dlc_practical_prologue import generate_pair_sets
from utils.metrics import accuracy
from utils.nn_utils import train_model

# Data related global variables
N = 1000  # Number of pairs

DATA_TENSORS = generate_pair_sets(N)

# Neural nets parameters
NN_ARGS = {"input_size": 14 * 14, "num_classes": 10}
BATCH_SIZE = 100
NUM_EPOCHS = 25


if __name__ == "__main__":
    # Datasets
    (
        train_input,
        train_target,
        train_classes,
        test_input,
        test_target,
        test_classes,
    ) = DATA_TENSORS

    train_dataset = data.TensorDataset(train_input, train_classes)
    test_dataset = data.TensorDataset(
        test_input, test_target
    )  # TODO: modify data loader so it returns a tuple with target and classes
    # Dataset Loader (Input Batcher)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    # Init a model
    linear_relu_model = linear_relu_net(**NN_ARGS)
    lenet = lenet()
    # Train a model
    train_model(linear_relu_model, NUM_EPOCHS, train_loader)
    train_model(lenet, NUM_EPOCHS, train_loader, flatten=False)
    # Test a model
    print("LinearRelu model accuracy:")
    print(accuracy(linear_relu_model, test_loader))
    print("LeNet model accuracy:")
    print(accuracy(lenet, test_loader, flatten=False))
