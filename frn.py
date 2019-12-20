import torch
import torch.nn as nn


class FilterResponseNormalization(nn.Module):
    
    def __init__(self, num_features, eps=1e-6, learnable_eps=False):
        """
        Input Variables:
        ----------------
            num_features: A integer indicating the number of input feature dimensions.
            eps: A scalar constant or learnable variable.
            learnable_eps: A bool value indicating whether the eps is learnable.
        """
        super(FRN, self).__init__()
        self.eps = nn.Parameter(torch.Tensor([eps]))
        if not learnable_eps:
            self.eps.requires_grad_(False)
        self.gamma = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.tau = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.reset_parameters()
    
    def forward(self, x):
        """
        Input Variables:
        ----------------
            x: Input tensor of shape [NxCxHxW]
        """
        nu2 = torch.pow(x, 2).mean(dim=(2, 3), keepdim=True)
        x = x * torch.rsqrt(nu2 + torch.abs(self.eps))
        return torch.max(self.gamma * x + self.beta, self.tau)

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        nn.init.zeros_(self.tau)
