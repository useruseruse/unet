import torch.optim as optim
from torch.utils.data import  DataLoader, random_split
from torchvision import transforms 
import torch 

from unet import UNet 
from dataset import CustomDataset
from dice_loss import dice_loss

# Initialize U-Net
model = UNet(n_channels=1, n_class=1)  # n_channel = 1  input is grey scale img
# DataLoader
image_directory = "/Users/namjihyeon/Desktop/hankooktier/unet/src/data/grey"
mask_directory = "/Users/namjihyeon/Desktop/hankooktier/unet/src/data/masks"

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.456], std=[0.224])
])


def train(model, optimizer, num_epochs):
    for epoch in range(num_epochs): 
        for image1, image2  in train_loader:
            # Process your images through the autoencoder
            # Choose image1 or image2 as input based on your requirement
            output_1 = model(image1)        
            output_2 = model(image2)

            # Compute loss
            loss = dice_loss(output_1, output_2)  # Or use image2, depending on your setup

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss}')
    
def test():
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for image_1, image_2  in test_loader:
            output_1 = model(image_1)
            output_2 = model(image_2)
            loss = dice_loss(output_1, output_2)
            # Here, you can print out loss, or store it for analysis
            print("loss",loss)


custom_dataset = CustomDataset(image_directory, mask_directory, transform=transform)  # Define your paths and transformations

train_size = int(0.8 * len(custom_dataset))  # 80% for training
test_size = len(custom_dataset) - train_size

train_dataset, test_dataset = random_split(custom_dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)


model = UNet(n_channels=1)
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1

train(model, optimizer, num_epochs)
test()