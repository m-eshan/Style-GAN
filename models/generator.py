import torch
import torch.nn as nn
from .mapping import MappingNetwork
from .blocks import StyleConvBlock

class StyleGenerator(nn.Module):
    """
    Synthesis Network starting from a learned constant tensor modulated progressively with AdaIN and noise injection.
    
    Args:
        latent_dim (int): Input noise & style vector dimension. Defaults to 128.
    """
    def __init__(self, latent_dim=128):
        super(StyleGenerator, self).__init__()
        self.mapping = MappingNetwork(latent_dim)
        
        # Initialize structural constant base tensor as ones so it is highly modifiable
        self.const = nn.Parameter(torch.ones(1, 128, 4, 4))
        
        # Progressive synthesis scales
        self.block1 = StyleConvBlock(128, 128, latent_dim)                                    # Scale 4x4
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)   # to 8x8
        self.block2 = StyleConvBlock(128, 64, latent_dim)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)   # to 16x16
        self.block3 = StyleConvBlock(64, 32, latent_dim)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)   # to 32x32
        self.block4 = StyleConvBlock(32, 3, latent_dim)
        self.to_rgb = nn.Tanh()

    def forward(self, z, style_mixing_step=None, z2=None):
        w = self.mapping(z)
        w2 = self.mapping(z2) if z2 is not None else w
        
        x = self.const.repeat(z.size(0), 1, 1, 1)
        
        # Add small spatial noise for stochastic variation to break mode collapse
        noise1 = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device) * 0.15
        x = x + noise1
        
        # Style Injection selection parameters
        w_early = w
        w_late = w2 if (style_mixing_step is not None and style_mixing_step <= 2) else w
        
        x = self.block1(x, w_early)
        x = self.upsample1(x)
        
        noise2 = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device) * 0.15
        x = x + noise2
        x = self.block2(x, w_early)
        
        w_late = w2 if (style_mixing_step is not None and style_mixing_step <= 4) else w
        x = self.upsample2(x)
        
        noise3 = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device) * 0.15
        x = x + noise3
        x = self.block3(x, w_late)
        x = self.upsample3(x)
        
        noise4 = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device) * 0.15
        x = x + noise4
        x = self.block4(x, w_late)
        
        return self.to_rgb(x)
