#!/usr/bin/env python3
"""
File: test.py
Description: Mini project 1 main file
"""
import torch
from torch.utils import data

from models.LeNet import lenet
from models.LinearReluNet import linear_relu_net
from models.ResNet import residual_net
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
MODELS = {
    "LinearRelu": {"args": NN_ARGS, "f": linear_relu_net, "flatten": True},
    "LeNet": {"args": None, "f": lenet, "flatten": False},
    "ResNet": {"args": None, "f": residual_net, "flatten": False},
}


def init_loaders():
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
    return train_loader, test_loader


def train_test(train_loader, test_loader):
    for key in MODELS:
        args = MODELS.get(key).get("args")
        f = MODELS.get(key).get("f")
        flatten = MODELS.get(key).get("flatten")
        name = key
        # Init
        if args is None:
            model = f()
        else:
            model = f(**args)

        print("Training " + name)
        # Train
        train_model(model, NUM_EPOCHS, train_loader, flatten=flatten)

        print("Testing " + name)
        # Testing
        print(accuracy(model, test_loader, flatten=flatten))


if __name__ == "__main__":

    # Prepare data
    train_loader, test_loader = init_loaders()

    # Train and test models
    train_test(train_loader, test_loader)
