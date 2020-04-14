"""
File: test.py
Description: Use case of the implemented framework
"""
import mini_project_2.fwk.neural_nets as nn
import mini_project_2.fwk.optimizers as optim
import torch
from collections import OrderedDict

torch.set_grad_enabled(False)


class CustomNet(nn.Module):

    def __init__(self):
        super(CustomNet, self).__init__()
        self.f = nn.Sequential(OrderedDict(
            {
                'linear 1': nn.Linear(2, 2),
                'tanh 1': nn.Tanh()
            }
        )
        )

    def backward(self, d_loss):
        return self.f.backward(d_loss)

    def forward(self, x):
        return self.f(x)


if __name__ == "__main__":
    # Dummy data
    x = torch.tensor([1., 2.])  # TODO: generate data from distribution
    y = torch.tensor([0., 3.])

    # Dummy use case
    model = CustomNet()
    criterion = nn.MSE()
    optimizer = optim.SGD(model.param())

    # Forward
    output = model(x)
    print(output)
    loss = criterion(output, y)

    # Backward
    loss.backward(model)

    # Update weights
    optimizer.step()

    # Print loss
    print(loss.item())
