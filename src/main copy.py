import os
import torch
from torchvision import transforms 

import torch.optim as optim

from model.unet import UNet 
from preprocess.dataset import *
from train import train, test

def load_model(model_path):
    # 모델 정의 (예: UNet)
    model = UNet(n_channels=1)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    if os.path.exists(model_path) :

        try:
            model.load_state_dict(torch.load(model_path))
            model.eval()
            print(" SUCCESS ㅅpretrained model ")
            return model, True
        except FileNotFoundError:
            print(" NO pretrained model ")
            return model, False

        
    else:
        train(model, optimizer, train_loader, checkpoint_path)
        print("Complete training")



# all 일 경우 grey shaped data 를, 
def main(dataset = "all", 
        image_directory = None, 
        mask_directory = None,
        checkpoint_path = None,
    ):
   

    # 1. 데이터셋이 준비되지 않았을 경우, 데이터셋을 준비. 
    if(mask_directory == None):
        mask_directory = ## create_mask(image_directory, redirect directory) 로 ground truth image mask 생성
        return 

    # 2. dataset 준비 
    train_loader, test_loader = prepare_dataloader(image_directory, mask_directory)
  
    # 3. pretrained 된 모델이 있을 경우 불러오기 
    model, success = load_pretrained_model()

    test(model, test_loader)  # Assuming 'test' function takes model and test_loader as arguments

if __name__ == "__main__":
    image_directory = "../data/redirect", 
    mask_directory = "../data/masks"
    checkpoint_path = './model_checkpoint.pth'  