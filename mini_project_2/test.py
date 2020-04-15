"""
File: test.py
Description: Use case of the implemented framework
"""
import mini_project_2.fwk.neural_nets as nn
import mini_project_2.fwk.optimizers as optim
import torch
from collections import OrderedDict

# Disable auto_grad
torch.set_grad_enabled(False)

# Global variables
N_EPOCHS = 100


class CustomNet(nn.Module):

    def __init__(self):
        super(CustomNet, self).__init__()
        self.f = nn.Sequential(OrderedDict(
            {
                'linear 1': nn.Linear(2, 25),
                'tanh 1': nn.Tanh(),
                'linear 2': nn.Linear(25, 25),
                'tanh 2': nn.Tanh(),
                'linear 3': nn.Linear(25, 25),
                'tanh 3': nn.Tanh(),
                'linear 4': nn.Linear(25, 2),
                'tanh 4': nn.Tanh(),
            }
        )
        )

    def backward(self, d_loss):
        return self.f.backward(d_loss)

    def forward(self, x):
        return self.f(x)


if __name__ == "__main__":
    # Dummy data
    X = [torch.tensor([0.5, 0.7]), torch.tensor([2., 1.])]  # TODO: generate data from distribution
    Y = [torch.tensor([1., 0.]), torch.tensor([0., 1.])]

    # Dummy use case
    model = CustomNet()
    criterion = nn.MSE()
    optimizer = optim.SGD(model.param(), eta=0.1/len(X))

    losses = []
    for i in range(N_EPOCHS):
        avg_loss = 0
        model.zero_grad()
        for n in range(len(X)):
            # Forward
            output = model(X[n])
            loss = criterion(output, Y[n])
            avg_loss += loss.item() / len(X)

            # Backward
            loss.backward(model)

        # Update weights
        optimizer.step()

        # Print loss
        print(avg_loss)
