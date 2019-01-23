import torch
import torch.nn.functional as F

def logsumexp(a, b):
    min_, max_ = torch.min(a, b), torch.max(a, b)
    return max_ + F.softplus(min_ - max_)
