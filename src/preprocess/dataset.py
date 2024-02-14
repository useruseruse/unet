from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as transforms
import os


class CustomDataset(Dataset):
    def __init__(self, directory, mask_directory, transform = None):
        if(transform == None):              # default transform 
            transform = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.456], std=[0.224])
        ])
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
            trans_masks = (self.transform(mask) > 0).float()

        return trans_img, trans_masks


def load_dataloader(image_directory, mask_directory):
    '''
        image directory 와 mask directory 를 받아서,
        dataset 을 구성한 뒤 test loader 와 train loader 를 반환
    '''
    custom_dataset = CustomDataset(image_directory, mask_directory, transform=None)
    
    sampled_size = int(0.15 * len(custom_dataset))
    train_size = int(0.8 *sampled_size)
    test_size = sampled_size - train_size
    
    train_dataset, test_dataset = random_split(custom_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    return train_loader, test_loader
