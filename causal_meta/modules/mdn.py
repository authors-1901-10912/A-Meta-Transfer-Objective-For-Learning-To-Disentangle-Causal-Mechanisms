import torch
import torch.nn as nn
import torch.nn.functional as F


class MDN(nn.Module):
    def __init__(self, n_hidden, n_gaussians):
        super(MDN, self).__init__()
        self.z_h = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.Tanh()
        )
        self.z_pi = nn.Linear(n_hidden, n_gaussians)
        self.z_mu = nn.Linear(n_hidden, n_gaussians)
        self.z_sigma = nn.Linear(n_hidden, n_gaussians)

    def forward(self, x):
        z_h = self.z_h(x)
        pi = F.softmax(self.z_pi(z_h), -1)
        mu = self.z_mu(z_h)
        sigma = torch.exp(self.z_sigma(z_h))
        return pi, mu, sigma


class GMM(nn.Module):
    def __init__(self, n_gaussians):
        super(GMM, self).__init__()
        self.n_gaussians = n_gaussians
        self.pi = torch.nn.Parameter(torch.ones(1, n_gaussians))
        self.mu = torch.nn.Parameter(torch.empty(1, n_gaussians).normal_())
        self.sigma = torch.nn.Parameter(torch.ones(1, n_gaussians))

    def forward(self, like):
        return (F.softmax(self.pi, dim=-1).expand(like.shape[0], self.pi.shape[-1]),
                self.mu.repeat(like.shape[0], 1),
                torch.exp(self.sigma).expand(like.shape[0], self.sigma.shape[-1]))


def mdn_nll(pi_mu_sigma, y, reduce=True):
    pi, mu, sigma = pi_mu_sigma
    m = torch.distributions.Normal(loc=mu, scale=sigma)
    log_prob_y = m.log_prob(y)
    log_prob_pi_y = log_prob_y + torch.log(pi)
    loss = -torch.logsumexp(log_prob_pi_y, dim=1)
    if reduce:
        return torch.mean(loss)
    else:
        return loss

def _legacy_mdn_nll(pi_mu_sigma, y, reduce=True):
    pi, mu, sigma = pi_mu_sigma
    m = torch.distributions.Normal(loc=mu, scale=sigma)
    loss = torch.exp(m.log_prob(y))
    loss = torch.sum(loss * pi, dim=1)
    loss = -torch.log(loss)
    if reduce:
        return torch.mean(loss)
    else:
        return loss