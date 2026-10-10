import os
import argparse
import torch
from models import StyleGenerator
from utils.visualize import generate_style_mixing_grid, generate_lerp_interpolation

def main():
    parser = argparse.ArgumentParser(description="StyleGAN Inference & Visualization Generator (Max 4 images per cell)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/netG.pth", help="Path to netG checkpoint")
    parser.add_argument("--latent_dim", type=int, default=128, help="Latent vector dimension")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Directory to save generated figures")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    netG = StyleGenerator(latent_dim=args.latent_dim).to(device)
    if os.path.exists(args.checkpoint):
        netG.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded weights from {args.checkpoint}")
    else:
        print("Checkpoint not found. Using initialized model.")

    netG.eval()
    os.makedirs(args.out_dir, exist_ok=True)

    print("1. Generating 2x2 Style Mixing Grid (Max 4 images)...")
    generate_style_mixing_grid(netG, device, args.latent_dim, save_path=os.path.join(args.out_dir, "exp9_style_mixing.png"))

    print("2. Generating Latent Space LERP Interpolation (4 Steps)...")
    generate_lerp_interpolation(netG, device, args.latent_dim, save_path=os.path.join(args.out_dir, "exp9_latent_interpolation.png"))

    print("All outputs generated successfully!")

if __name__ == "__main__":
    main()
