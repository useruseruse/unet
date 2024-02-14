import torch
from torchvision import transforms
from PIL import Image
from torch.nn.functional import cosine_similarity
from model.unet import UNet
import torch.optim as optim
from model.dice_loss import *
# 모델 불러오기
def load_model(checkpoint_path):
    model = UNet(n_channels=1) # to be fixed
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    return model

# 이미지 전처리
def transform_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.456], std=[0.224])
    ])
    return transform(image).unsqueeze(0)  # 이미지를 tensor로 변환하고, batch dimension 추가

# Latent Vector 추출
def get_latent_vector(model, image_tensor):
    with torch.no_grad():
        latent_vector = model.get_latent_vector(image_tensor)
        return latent_vector
    
def flatten_tensor(tensor):
    return tensor.view(tensor.size(0), -1)  # Flatten the tensor

def adaptive_pool(tensor):
    pool = torch.nn.AdaptiveAvgPool2d(1)
    return pool(tensor).view(tensor.size(0), -1)  # Pool and then flatten


# 메인 함수
def calc_cosine(image_path1, image_path2, model_path):
   
    model = load_model(model_path)
    print("loaded model")

    image1 = transform_image(image_path1)
    image2 = transform_image(image_path2)

    # latent vector 의 사이즈는 16 x 16 x 512 
    latent_vector1 = get_latent_vector(model, image1)
    latent_vector2 = get_latent_vector(model, image2)
    latent_vector1 =model(image1)
    latent_vector2 = model(image2)
    
    dice = dice_loss(latent_vector1, latent_vector2)
    print(dice)

    # Apply adaptive pooling
    vector1_pooled = adaptive_pool(latent_vector1)
    vector2_pooled = adaptive_pool(latent_vector2)

    similarity = torch.nn.functional.cosine_similarity(vector1_pooled, vector2_pooled, dim=1)
    print(f'Cosine Similarity (Adaptive Pooling): {similarity.item()}')

    # 1. 이를 flatten [0,-1] 하여 sim 구함
    vector1_flat = flatten_tensor(latent_vector1)
    vector2_flat = flatten_tensor(latent_vector2)

    similarity = torch.nn.functional.cosine_similarity(vector1_flat, vector2_flat, dim=1)
    print(f'Cosine Similarity (Flatten): {similarity.item()}')


if __name__ == "__main__":
    image_path1 = "../data_result/redirect/grey_20OE010370KI_1_100.jpg_generation.png"
    # image_path2 = "../data_result/redirect/grey_20OE010370KI_1_100.jpg.png"
    image_path2="../data_result/redirect/grey_20RS010169KI_4_100.jpg_generation.png"
    path_to_model = "./model_cache/model_checkpoint_29.pth"
    calc_cosine(image_path1, image_path2, path_to_model)
