import torch
import torch.nn as nn

class FilterResponseNormalization(nn.Module):
    def __init__(self, beta, gamma, tau, eps=1e-6):
        # x: Input tensor of shape [NxCxHxW].
        # tau, beta, gamma: Variables of shape [1, C, 1, 1].
        # eps: A scalar constant or learnable variable.

        super(FilterResponseNormalization, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.eps = torch.Tensor([eps])

    def forward(self, x):
        n, c, h, w = x.shape
        assert (gamma.shape[1], beta.shape[1], tau.shape[1]) == (c, c, c)

        nu2 = torch.mean(x.pow(2), (2,3), keepdims=True)
        x = torch.div(x, nu2+torch.abs(self.eps))
        return torch.max(self.gamma*x + self.beta, self.tau)