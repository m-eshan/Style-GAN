import torch
import torch.nn as nn

class MappingNetwork(nn.Module):
    """
    Maps random noise vector z to disentangled intermediate latent space w via a robust 4-layer MLP.
    
    Args:
        latent_dim (int): Dimension of input noise z and output style vector w. Defaults to 128.
    """
    def __init__(self, latent_dim=128):
        super(MappingNetwork, self).__init__()
        layers = []
        # Reduced from 8 layers to 4 layers to avoid premature representational collapse
        for _ in range(4):
            layers.append(nn.Linear(latent_dim, latent_dim))
            layers.append(nn.LeakyReLU(0.2))
        self.mapping = nn.Sequential(*layers)

    def forward(self, z):
        return self.mapping(z)
