import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain

from causal_meta.utils.torch_utils import logsumexp

class BinaryStructuralModel(nn.Module):
    def __init__(self, model_A_B, model_B_A):
        super(BinaryStructuralModel, self).__init__()
        self.model_A_B = model_A_B
        self.model_B_A = model_B_A
        self.w = nn.Parameter(torch.tensor(0., dtype=torch.float64))

    def forward(self, inputs):
        return self.online_loglikelihood(self.model_A_B(inputs), self.model_B_A(inputs))

    def online_loglikelihood(self, logl_A_B, logl_B_A):
        n = logl_A_B.size(0)
        log_alpha, log_1_m_alpha = F.logsigmoid(self.w), F.logsigmoid(-self.w)

        return logsumexp(log_alpha + torch.sum(logl_A_B),
            log_1_m_alpha + torch.sum(logl_B_A))# / float(n)

    def modules_parameters(self):
        return chain(self.model_A_B.parameters(), self.model_B_A.parameters())

    def structural_parameters(self):
        return [self.w]

class ModelA2B(nn.Module):
    def __init__(self, marginal, conditional):
        super(ModelA2B, self).__init__()
        self.p_A = marginal
        self.p_B_A = conditional

    def forward(self, inputs):
        inputs_A, inputs_B = torch.split(inputs, 1, dim=1)
        return self.p_A(inputs_A) + self.p_B_A(inputs_B, inputs_A)

class ModelB2A(nn.Module):
    def __init__(self, marginal, conditional):
        super(ModelB2A, self).__init__()
        self.p_B = marginal
        self.p_A_B = conditional

    def forward(self, inputs):
        inputs_A, inputs_B = torch.split(inputs, 1, dim=1)
        return self.p_B(inputs_B) + self.p_A_B(inputs_A, inputs_B)
