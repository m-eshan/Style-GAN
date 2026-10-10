import torch
import torch.nn as nn
from .adain import AdaIN

class StyleConvBlock(nn.Module):
    """
    Standard Convolutional step embedded with independent AdaIN layers for progressive style injection.
    
    Args:
        in_channels (int): Input feature map channels.
        out_channels (int): Output feature map channels.
        latent_dim (int): Dimension of style vector w. Defaults to 128.
    """
    def __init__(self, in_channels, out_channels, latent_dim=128):
        super(StyleConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.adain = AdaIN(out_channels, latent_dim)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, w):
        x = self.conv(x)
        x = self.act(x)
        x = self.adain(x, w)
        return x

class MinibatchStdDev(nn.Module):
    """
    Appends standard deviation of the batch across channels to force batch diversity and prevent mode collapse.
    """
    def forward(self, x):
        batch_size, _, height, width = x.shape
        # Compute average standard deviation over all features and pixels
        std = torch.std(x, dim=0, keepdim=True) + 1e-8
        mean_std = torch.mean(std).view(1, 1, 1, 1)
        # Tile and append as a new channel
        mean_std = mean_std.repeat(batch_size, 1, height, width)
        return torch.cat([x, mean_std], dim=1)
