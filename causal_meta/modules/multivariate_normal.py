import torch
import torch.nn as nn
import scipy

from torch.distributions import MultivariateNormal

class Marginal(nn.Module):
    def __init__(self, dim=1, dtype=None):
        super(Marginal, self).__init__()
        self.mean = nn.Parameter(torch.zeros((dim,), dtype=dtype))
        self.L = nn.Parameter(torch.ones((dim, dim), dtype=dtype))

    def forward(self, inputs):
        dist = MultivariateNormal(self.mean, scale_tril=torch.tril(self.L))
        log_prob = dist.log_prob(inputs)
        return log_prob

    def get_cov(self):
        L = torch.tril(self.L)
        return torch.matmul(L, torch.t(L))

    def set_cov(self, cov):
        L = scipy.linalg.cholesky(cov, lower=True)
        self.L.copy_(torch.from_numpy(L))


class Conditional(nn.Module):
    def __init__(self, dim=1, dtype=None):
        super(Conditional, self).__init__()
        self.linear = nn.Linear(dim, dim).to(dtype=dtype)
        self.L = nn.Parameter(torch.ones((dim, dim), dtype=dtype))

    def forward(self, inputs, conds):
        mean = self.linear(conds)
        dist = MultivariateNormal(mean, scale_tril=torch.tril(self.L))
        log_prob = dist.log_prob(inputs)
        return log_prob

    def get_cov(self):
        L = tril(self.L)
        return torch.matmul(L, torch.t(L))

    def set_cov(self, cov):
        L = scipy.linalg.cholesky(cov, lower=True)
        self.L.copy_(torch.from_numpy(L))
