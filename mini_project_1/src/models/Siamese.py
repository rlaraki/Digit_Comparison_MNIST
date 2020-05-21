from collections import OrderedDict

import torch.nn as nn
import torch


class Siamese(nn.Module):
    def __init__(self, child_model, flatten, is_weight_sharing=True):
        super(Siamese, self).__init__()
        self.flatten = flatten
        self.m1 = child_model
        self.ws = is_weight_sharing
        if not is_weight_sharing:
            self.m2 = child_model

        self.linear_ws = nn.Sequential(
            OrderedDict(
                [("F1", nn.Linear(20, 100)),
                 ("Relu1", nn.ReLU()),
                 ("F2", nn.Linear(100, 2)),
                 ("LogSoftmax", nn.LogSoftmax(dim=-1))]))

    def forward(self, x):

        x1 = x[:, 0, :, :]
        x2 = x[:, 1, :, :]

        if self.flatten:
            x1 = x1.view(-1, x1.shape[1] * x1.shape[2])
            x2 = x2.view(-1, x2.shape[1] * x2.shape[2])
        else:
            x1 = x1.unsqueeze(1)
            x2 = x2.unsqueeze(1)

        out1 = self.m1(x1)
        out2 = self.m1(x2) if self.ws else self.m2(x2)

        tot = torch.cat((out1, out2), 1)  # 10*20
        res = self.linear_ws(tot)
        return res, out1, out2


def siamese(child_model, flatten, weight_sharing):
    model = Siamese(child_model=child_model, flatten=flatten, is_weight_sharing=weight_sharing)
    return model

