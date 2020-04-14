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
        # TODO : create sequential

    def backward(self, d_loss):
        dl_x1 = self.tanh1.backward(d_loss)
        dl_s1 = self.linear1.backward(dl_x1)

        grads = []
        grads.append(dl_s1.view(-1, 1).mm(self.x.view(1, -1)))
        grads.append(dl_s1)
        self.add_parameter('grads', grads)

    def forward(self, x):
        return self.f(x)
#        self.x = x
#        s1 = self.linear1(x)
#        self.x1 = self.tanh1(s1)
#        return self.x1


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
