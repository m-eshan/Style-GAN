# #E0324009 - Eshan M
import os
import torch
import torch.nn as nn
import torch.optim as optim

from models import StyleGenerator, Discriminator
from data.dataset import get_dataloader
from utils.visualize import generate_style_mixing_grid, generate_lerp_interpolation

def weights_init(m):
    classname = m.__class__.__name__
    # Only apply weight initialization to leaf layers that actually contain weight parameters
    if hasattr(m, 'weight') and m.weight is not None:
        if classname.find('Conv') != -1 or classname.find('Linear') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0)

def train():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using execution device environment: {device}")

    # DataLoader
    batch_size = 64
    dataloader = get_dataloader(data_dir='./data', batch_size=batch_size, download=True)
    print(f"Dataset completely initialized. Batches available: {len(dataloader)}")

    # Models
    latent_dim = 128
    netG = StyleGenerator(latent_dim=latent_dim).to(device)
    netD = Discriminator().to(device)

    # Initialize weights
    netG.apply(weights_init)
    netD.apply(weights_init)
    print("Weights successfully initialized!")

    # Criterion & Optimizers
    criterion = nn.BCEWithLogitsLoss()
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=0.0001, betas=(0.5, 0.999))
    print("Networks successfully re-instantiated with MinibatchStdDev integration!")

    epochs = 5
    print("Starting Stabilized Adversarial Training Loop...")

    for epoch in range(epochs):
        for i, (real_imgs, _) in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            current_batch_size = real_imgs.size(0)

            # ---------------------
            # Train Discriminator
            # ---------------------
            optimizerD.zero_grad()
            z = torch.randn(current_batch_size, latent_dim).to(device)
            fake_imgs = netG(z).detach()

            # Label smoothing for real images (0.9 instead of 1.0)
            real_labels = torch.full((current_batch_size, 1), 0.9, device=device)
            fake_labels = torch.zeros((current_batch_size, 1), device=device)

            loss_real = criterion(netD(real_imgs), real_labels)
            loss_fake = criterion(netD(fake_imgs), fake_labels)
            loss_D = loss_real + loss_fake
            loss_D.backward()
            optimizerD.step()

            # ---------------------
            # Train Generator
            # ---------------------
            optimizerG.zero_grad()
            z = torch.randn(current_batch_size, latent_dim).to(device)
            gen_imgs = netG(z)

            # Generator wants to fool discriminator into thinking fakes are real
            loss_G = criterion(netD(gen_imgs), torch.ones((current_batch_size, 1), device=device))
            loss_G.backward()
            optimizerG.step()

        print(f"[Epoch {epoch+1}/{epochs}] Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f}")

    # Save trained model weights
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(netG.state_dict(), 'checkpoints/netG.pth')
    torch.save(netD.state_dict(), 'checkpoints/netD.pth')
    print("Saved model checkpoints to checkpoints/")

    # Generate Evaluation Outputs (Max 4 images per grid)
    print("Generating evaluation figures (Max 4 images per grid)...")
    generate_style_mixing_grid(netG, device, latent_dim, save_path='outputs/exp9_style_mixing.png')
    generate_lerp_interpolation(netG, device, latent_dim, save_path='outputs/exp9_latent_interpolation.png')
    print("Evaluation completed successfully!")

if __name__ == "__main__":
    train()
