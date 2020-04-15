"""
File: test.py
Description: Use case of the implemented framework
"""
import mini_project_2.fwk.neural_nets as nn
import mini_project_2.fwk.optimizers as optim
import mini_project_2.fwk.data as data
import torch
from collections import OrderedDict

# Disable auto_grad
torch.set_grad_enabled(False)

# Global variables
N_EPOCHS = 1000


class CustomNet(nn.Module):

    def __init__(self):
        super(CustomNet, self).__init__()
        self.f = nn.Sequential(OrderedDict(
            {
                'linear 1': nn.Linear(2, 25),
                'relu 1': nn.ReLU(),
                'linear 2': nn.Linear(25, 25),
                'relu 2': nn.ReLU(),
                'linear 3': nn.Linear(25, 25),
                'relu 3': nn.ReLU(),
                'linear 4': nn.Linear(25, 2),
                'last act': nn.Tanh(),
            }
        )
        )

    def backward(self, d_loss):
        return self.f.backward(d_loss)

    def forward(self, x):
        return self.f(x)


if __name__ == "__main__":
    # Data
    X, Y = data.generate_data()

    # Use case
    model = CustomNet()
    criterion = nn.MSE()

    eta = 0.001 / len(X)  # learning rate
    optimizer = optim.SGD(model.param(), eta=eta)

    losses = []
    step = len(X)
    print("#" * 10 + 'Train' + "#" * 10)
    for i in range(N_EPOCHS):
        avg_loss = 0
        error = 0
        model.zero_grad()
        for n in range(step):
            # Forward
            output = model(X[n])
            loss = criterion(output, Y[n])
            avg_loss += loss.item() / step

            # Compute error
            if output.max(0)[1].item() != Y[n].max(0)[1].item():
                error += 1

            # Backward
            loss.backward(model)

        # Update weights
        optimizer.step()
        # Decrease step size
        optimizer.set_eta(eta / (1 + (2 / (i + 1))))

        # Print loss
        print("Epoch %d, Loss=%.4f, Training_Acc=%.4f" % (i + 1, avg_loss, (step - error) / step))

    # Test
    print("#" * 10 + 'Test' + "#" * 10)
    # Data
    X_test, Y_test = data.generate_data()
    test_error = 0
    avg_test_loss = 0
    for i in range(len(X_test)):
        # Forward
        output = model(X_test[i])
        loss = criterion(output, Y_test[i])
        avg_test_loss += loss.item() / len(X_test)

        # Compute error
        if output.max(0)[1].item() != Y[i].max(0)[1].item():
            test_error += 1

    print("Loss=%.4f, Test_Acc=%.4f" % (avg_test_loss, (step - test_error) / step))
