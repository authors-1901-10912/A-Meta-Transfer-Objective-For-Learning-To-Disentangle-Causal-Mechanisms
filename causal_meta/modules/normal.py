import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Marginal(nn.Module):
    def __init__(self, dtype=None):
        super(Marginal, self).__init__()
        self.mean = nn.Parameter(torch.tensor(0., dtype=dtype))
        self.log_std = nn.Parameter(torch.tensor(0., dtype=dtype))

    def forward(self, inputs):
        cste = -0.5 * math.log(2 * math.pi) - self.log_std
        var = torch.exp(2 * self.log_std)
        log_normal = cste - F.mse_loss(self.mean, inputs, reduction='none') / (2 * var)
        return log_normal.squeeze(1)

    def init_parameters(self):
        self.mean.data.zero_()
        self.log_std.data.zero_()

class Conditional(nn.Module):
    def __init__(self, dtype=None):
        super(Conditional, self).__init__()
        self.linear = nn.Linear(1, 1).to(dtype=dtype)
        self.log_std = nn.Parameter(torch.tensor(0., dtype=dtype))

    def forward(self, inputs, conds):
        cste = -0.5 * math.log(2 * math.pi) - self.log_std
        var = torch.exp(2 * self.log_std)
        log_normal = cste - F.mse_loss(self.linear(conds), inputs, reduction='none') / (2 * var)
        return log_normal.squeeze(1)

    def init_parameters(self):
        self.linear.reset_parameters()
        self.log_std.data.zero_()
