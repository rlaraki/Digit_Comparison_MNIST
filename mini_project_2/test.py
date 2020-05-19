"""
File: test.py
Description: Use case of the implemented framework
"""

# Import projects libs
import fwk.neural_nets as nn
import fwk.optimizers as optim
from utils import *


import torch
from collections import OrderedDict


# Disable auto_grad
torch.set_grad_enabled(False)

# Global variables
N_EPOCHS = 100


# Create example feedforward neural network
class CustomNet(nn.Module):

    def __init__(self):
        self.f = nn.Sequential(OrderedDict(
            {
                'linear 1': nn.Linear(2, 25),
                'relu 1': nn.Tanh(),
                'linear 2': nn.Linear(25, 25),
                'relu 2': nn.Tanh(),
                'linear 3': nn.Linear(25, 25),
                'relu 3': nn.Tanh(),
                'linear 4': nn.Linear(25, 2),
                'last act': nn.Tanh(),
            }
        )
        )

    def backward(self, d_loss):
        return self.f.backward(d_loss)

    def forward(self, x):
        return self.f.forward(x)
    
    def zero_grad(self):
        return self.f.zero_grad()
    
    def param(self):
        return self.f.param()

if __name__ == "__main__":
    
    # Generate train and test set. 
    X_train, Y_train = generate_data()
    X_test, Y_test = generate_data()
    
    # Init training parameters
    model = CustomNet()
    optimizer = optim.SGD(model.param())
    criterion = nn.MSE()
    
    # Train model
    print("--- Training ---")
    train(X_train, Y_train, model, optimizer, criterion, n_epochs=N_EPOCHS, verbose=True)
    
    print("--- Testing ---")
    
    # Test model accuracy
    print("Train accuracy:", accuracy(model, X_train, Y_train))
    print("Test accuracy:", accuracy(model, X_test, Y_test))
    
   
