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
        self.linear = nn.Linear()

    def backward(self, out):
        return out  # Dummy

    def forward(self, input):
        x = self.linear(input)  # Is None for the moment
        return x


if __name__ == "__main__":
    # Dummy data
    x = torch.tensor([1, 2, 3])
    y = torch.tensor([1])

    # Dummy use case
    model = CustomNet()
    criterion = nn.MSE()
    optimizer = optim.SGD(model.param())

    # Forward
    output = model(x)
    loss = criterion(output, y)

    # Backward
    loss.backward()

    # Update weights
    optimizer.step()

    # Print loss
    print(loss.item())





