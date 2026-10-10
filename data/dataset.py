import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_dataloader(data_dir='./data', batch_size=64, download=True):
    """
    Constructs FashionMNIST DataLoader formatted for 32x32 RGB StyleGAN training.
    
    Args:
        data_dir (str): Root directory to store dataset.
        batch_size (int): Batch size. Defaults to 64.
        download (bool): Download if not present. Defaults to True.
        
    Returns:
        DataLoader: PyTorch DataLoader for FashionMNIST.
    """
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),                        # Tile 1-channel to 3 channels (RGB simulation)
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))                  # Standard normalisation scale [-1, 1]
    ])

    dataset = torchvision.datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=download,
        transform=transform
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader
