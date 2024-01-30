import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self, width=32):
        super(SimpleNet, self).__init__()
        self.lin1 = nn.Linear(1, width)
        self.lin2 = nn.Linear(width, width)
        self.lin3 = nn.Linear(width, 1)
        self.act1 = nn.Softplus()
        self.act2 = nn.Softplus()

    def forward(self, x):
        fx = self.act1(self.lin1(x))
        fx = self.act2(self.lin2(fx))
        fx = self.lin3(fx)
        return {"x": x, "fx": fx}


def get_model(args):
    if args.model == "simple":
        return SimpleNet()
    else:
        raise NameError("Model type not recognized.")
