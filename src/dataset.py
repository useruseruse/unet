import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms 
from PIL import Image
import os
import cv2

class CustomDataset(Dataset):
    def __init__(self, directory, mask_directory, transform = None):
        self.transform = transform
        self.img_paths = [os.path.join(directory, file) for file in os.listdir(directory)]
        self.mask_paths = [os.path.join(mask_directory, file) for file in os.listdir(mask_directory)]
    
    def __len__(self):
        return (len(self.img_paths))
    
    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx])
        mask = Image.open(self.mask_paths[idx])

        if self.transform:
            trans_img = self.transform(image)
            trans_masks = self.transform(mask)

        return trans_img, trans_masks


