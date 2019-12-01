import torch
import torch.nn as nn

class FilterResponseNormalization(nn.Module):
    def __init__(self, beta, gamma, tau, eps=1e-6):
        # x: Input tensor of shape [NxCxHxW].
        # beta, gamma, tau: Variables of shape [1, C, 1, 1].
        # eps: A scalar constant or learnable variable.

        super(FilterResponseNormalization, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.eps = torch.Tensor([eps])

    def forward(self, x):
        n, c, h, w = x.shape
        assert (self.gamma.shape[1], self.beta.shape[1], self.tau.shape[1]) == (c, c, c)

        nu2 = torch.mean(x.pow(2), (2,3), keepdims=True)
        x = x * torch.rsqrt(nu2 + torch.abs(self.eps))
        return torch.max(self.gamma*x + self.beta, self.tau)