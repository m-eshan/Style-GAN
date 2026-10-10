import os
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for clean server figure rendering
import matplotlib.pyplot as plt
import torch
import numpy as np

def generate_style_mixing_grid(netG, device, latent_dim=128, save_path='outputs/exp9_style_mixing.png'):
    """
    Generates a ultra-clean 2x2 Style Mixing Grid (Max 4 images per cell/plot).
    Top-Left: Source A | Top-Right: Source B
    Bottom-Left: Mixed (A early + B late) | Bottom-Right: Mixed (B early + A late)
    """
    netG.eval()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with torch.no_grad():
        z1 = torch.randn(1, latent_dim).to(device)
        z2 = torch.randn(1, latent_dim).to(device)

        # Generate 4 distinct images
        img_a = netG(z1)
        img_b = netG(z2)
        mix1 = netG(z1, style_mixing_step=2, z2=z2)
        mix2 = netG(z2, style_mixing_step=2, z2=z1)

        images = [img_a, img_b, mix1, mix2]
        titles = [
            "Source Style A (w1)",
            "Source Style B (w2)",
            "Mixed: Style A + Style B",
            "Mixed: Style B + Style A"
        ]

        fig, axes = plt.subplots(2, 2, figsize=(8, 8), dpi=150)
        
        for idx, (img, title) in enumerate(zip(images, titles)):
            r, c = idx // 2, idx % 2
            ax = axes[r, c]
            
            img_np = (img.cpu().squeeze().permute(1, 2, 0).numpy() + 1) / 2
            ax.imshow(np.clip(img_np, 0, 1))
            ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
            ax.axis('off')

        plt.suptitle("StyleGAN Style Mixing (4 Sample Grid)", y=0.98, fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
        plt.close()

def generate_lerp_interpolation(netG, device, latent_dim=128, save_path='outputs/exp9_latent_interpolation.png'):
    """
    Generates Latent Vector LERP Interpolation Sequence with exactly 4 clean steps (1x4 grid).
    """
    netG.eval()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with torch.no_grad():
        z1 = torch.randn(1, latent_dim).to(device)
        z2 = torch.randn(1, latent_dim).to(device)

        steps = 4 # Strictly max 4 images per cell/plot
        alphas = np.linspace(0, 1, steps)
        
        fig, axes = plt.subplots(1, steps, figsize=(12, 3.5), dpi=150)

        for idx, alpha in enumerate(alphas):
            # Apply Linear Interpolation calculation sequence (LERP)
            z_interp = (1 - alpha) * z1 + alpha * z2
            interp_img = netG(z_interp)
            interp_np = (interp_img.cpu().squeeze().permute(1, 2, 0).numpy() + 1) / 2

            axes[idx].imshow(np.clip(interp_np, 0, 1))
            axes[idx].set_title(f"α = {alpha:.2f}", fontsize=12, fontweight='bold', pad=6)
            axes[idx].axis('off')

        plt.suptitle("Latent Space LERP Interpolation (4 Steps)", y=1.05, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
        plt.close()
