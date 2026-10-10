import os
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for server/script figure saving
import matplotlib.pyplot as plt
import torch
import numpy as np

def generate_style_mixing_grid(netG, device, latent_dim=128, save_path='outputs/exp9_style_mixing.png'):
    """
    Generates a 5x5 Style Mixing Grid Matrix Output.
    """
    netG.eval()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with torch.no_grad():
        z_source_a = torch.randn(4, latent_dim).to(device) # Source Row Coordinates
        z_source_b = torch.randn(4, latent_dim).to(device) # Source Column Coordinates

        fig, axes = plt.subplots(5, 5, figsize=(10, 10))

        # Generate Row Source Frames
        for c in range(4):
            img_b = netG(z_source_b[c:c+1])
            img_b = (img_b.cpu().squeeze().permute(1, 2, 0).numpy() + 1) / 2
            axes[0, c+1].imshow(np.clip(img_b, 0, 1))
            axes[0, c+1].axis('off')

        # Generate Column Source Frames
        for r in range(4):
            img_a = netG(z_source_a[r:r+1])
            img_a = (img_a.cpu().squeeze().permute(1, 2, 0).numpy() + 1) / 2
            axes[r+1, 0].imshow(np.clip(img_a, 0, 1))
            axes[r+1, 0].axis('off')

        # Generate Cross-Mixed Matrices
        for r in range(4):
            for c in range(4):
                mixed_img = netG(z_source_a[r:r+1], style_mixing_step=2, z2=z_source_b[c:c+1])
                mixed_img = (mixed_img.cpu().squeeze().permute(1, 2, 0).numpy() + 1) / 2
                axes[r+1, c+1].imshow(np.clip(mixed_img, 0, 1))
                axes[r+1, c+1].axis('off')

        axes[0, 0].axis('off')
        plt.suptitle("Style Mixing Grid Matrix Output", y=0.95, fontsize=14)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

def generate_lerp_interpolation(netG, device, latent_dim=128, steps=8, save_path='outputs/exp9_latent_interpolation.png'):
    """
    Generates Latent Vector LERP Interpolation Sequence across 8 steps.
    """
    netG.eval()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with torch.no_grad():
        z1 = torch.randn(1, latent_dim).to(device)
        z2 = torch.randn(1, latent_dim).to(device)

        fig, axes = plt.subplots(1, steps, figsize=(16, 3))

        for idx, alpha in enumerate(np.linspace(0, 1, steps)):
            # Apply Linear Interpolation calculation sequence (LERP)
            z_interp = (1 - alpha) * z1 + alpha * z2
            interp_img = netG(z_interp)
            interp_img = (interp_img.cpu().squeeze().permute(1, 2, 0).numpy() + 1) / 2

            axes[idx].imshow(np.clip(interp_img, 0, 1))
            axes[idx].set_title(f"α = {alpha:.2f}")
            axes[idx].axis('off')

        plt.suptitle("Latent Vector LERP Interpolation Sequence", y=1.05, fontsize=14)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
