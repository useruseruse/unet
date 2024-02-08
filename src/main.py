import os
import torch
from torchvision import transforms 
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

from model.unet import UNet 
from preprocess.dataset import CustomDataset
from train import train, test

if __name__ == "__main__":
    # DataLoader
    image_directory = "../data/redirect"
    mask_directory = "../data/masks"

    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.456], std=[0.224])
    ])

    custom_dataset = CustomDataset(image_directory, mask_directory, transform=transform)
    train_size = int(0.8 * len(custom_dataset))
    test_size = len(custom_dataset) - train_size
    train_dataset, test_dataset = random_split(custom_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    print("Complete loading dataset")

    model = UNet(n_channels=1)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    checkpoint_path = './model_checkpoint.pth'  # Path to your checkpoint file

    if os.path.exists(checkpoint_path) :
        # Load the saved model and optimizer state
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Loaded saved model and optimizer state.")
    else:
        # Train the model
        num_epochs = 17
        train(model, optimizer, num_epochs, train_loader, checkpoint_path)
        print("Complete training")

    test(model, test_loader)  # Assuming 'test' function takes model and test_loader as arguments
    print("Test model")
