import torch
import torch.nn as nn
from .blocks import MinibatchStdDev

class Discriminator(nn.Module):
    """
    Standard CNN classification network verifying macro structural realness frames with Batch Diversity check.
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),   # 32x32 -> 16x16
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 16x16 -> 8x8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 8x8 -> 4x4
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            MinibatchStdDev(),                                     # Appends diversity statistic channel
            nn.Conv2d(129, 128, kernel_size=3, padding=1),          # Process appended diversity channel
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1)
        )

    def forward(self, img):
        return self.model(img)
