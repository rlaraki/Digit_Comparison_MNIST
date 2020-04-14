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
        self.tanh1 = nn.Tanh()
        self.linear2 = nn.Linear(25, 2)

    def backward(self, d_loss):
        dl_s2 = self.linear2.backward(d_loss)
        dl_x1 = self.tanh1.backward(dl_s2)
        dl_s1 = self.linear1.backward(dl_x1)

        weights = []
        weights.append(dl_s2.view(-1, 1).mm(self.x1.view(1, -1)))
        weights.append(dl_s1.view(-1, 1).mm(self.x.view(1, -1)))
        self.add_parameter('weights', weights)


    def forward(self, x):
        self.x = x
        s1 = self.linear1(x)
        self.x1 = self.tanh1(s1)
        s2 = self.linear2(self.x1)
        return s2


if __name__ == "__main__":
    # Dummy data
    x = torch.tensor([1., 2.]) # TODO: generate data from distribution
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
    loss.backward(model)

    # Update weights
    optimizer.step()

    # Print loss
    print(loss.item())





