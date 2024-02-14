import os
import torch
import torch.optim as optim
from preprocess.create_mask import *

from model.unet import UNet 
from preprocess.dataset import *
from train import train, test

def load_model(model, optimizer, model_path):
    '''
        pretrained model 을 성공적으로 로드했을 경우 반환
    '''
    try:
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print(" SUCCESS to LOAD pretrained model ")
        return model, True
    
    except:
        print(" NO pretrained model ")
        return model, False

def main( 
        image_directory = None, 
        mask_directory = None,
        model_path = None,
    ):
   

    # 1. mask 이미지 셋이 준비되지 않았을 경우. 
    if(os.path.exists(mask_directory)):
        mask_directory = create_masks_from_directory(image_directory)
        print("mask dataset is saved", mask_directory)    

    # 2. dataset 준비 
    train_loader, test_loader = load_dataloader(image_directory, mask_directory)
    
    # 3. pretrained 된 모델이 있을 경우 불러오기 
    unet = UNet(n_channels=1)                               # 가중치를 덮어쓸 모델 기본 아키텍쳐 불러오기
    optimizer = optim.Adam(unet.parameters(), lr=0.0005)
    model, success = load_model(unet, optimizer, model_path) 

    # 4. pretrained 된 모델이 없을 경우, 새로 train 하기
    if not success:
        train(model, optimizer, train_loader, num_epochs=5)



if __name__ == "__main__":
    image_directory = "../data_grey/redirect"
    mask_directory = "../data_grey/masks"
    # checkpoint_path = './model_checkpoint.pth'  
    main(image_directory=image_directory, mask_directory=mask_directory)
    # mask_directory=mask_directory)