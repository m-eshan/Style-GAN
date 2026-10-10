import torch
import torch.nn as nn

class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization Layer scaling and shifting channels using style vectors w.
    
    Args:
        channels (int): Number of feature map channels.
        latent_dim (int): Dimension of style vector w. Defaults to 128.
    """
    def __init__(self, channels, latent_dim=128):
        super(AdaIN, self).__init__()
        self.instance_norm = nn.InstanceNorm2d(channels, affine=False)
        self.style_scale = nn.Linear(latent_dim, channels)
        self.style_shift = nn.Linear(latent_dim, channels)

    def forward(self, x, w):
        x_norm = self.instance_norm(x)
        gamma = self.style_scale(w).unsqueeze(-1).unsqueeze(-1)
        beta = self.style_shift(w).unsqueeze(-1).unsqueeze(-1)
        return gamma * x_norm + beta
