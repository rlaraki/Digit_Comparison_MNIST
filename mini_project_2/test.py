"""
File: test.py
Description: Use case of the implemented framework
"""
import mini_project_2.fwk.neural_nets as nn
import mini_project_2.fwk.optimizers as optim
import torch

torch.set_grad_enabled(False)


class CustomNet(nn.Module):

    def __init__(self):
        super(CustomNet, self).__init__()
        self.linear1 = nn.Linear(2, 25)
        self.linear2 = nn.Linear(25, 25)
        self.linear3 = nn.Linear(25, 25)
        self.linear4 = nn.Linear(25, 2)

    def backward(self, out):
        return out  # Dummy

    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.linear2(x1)
        x3 = self.linear3(x2)
        x4 = self.linear4(x3)
        return x4


if __name__ == "__main__":
    # Dummy data
    x = torch.tensor([1., 3.]) # TODO: generate data from distribution
    y = torch.tensor([1.])

    # Dummy use case
    model = CustomNet()
    criterion = nn.MSE()
    optimizer = optim.SGD(model.param())

    # Forward
    output = model(x)
    print(output)
    loss = criterion(output, y)

    # Backward
    loss.backward()

    # Update weights
    optimizer.step()

    # Print loss
    print(loss.item())





