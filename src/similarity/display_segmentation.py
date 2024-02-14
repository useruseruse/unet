import torch
import time
import matplotlib.pyplot as plt
from cosine import *



def display_image(image):
    plt.imshow(image, cmap='gray')
    plt.axis('off') 
    plt.show()
    time.sleep(30)



def display_segmentation(output_tensor):
    '''
        Unet model 의 output tensor (1,1,128,128) 를 입력받은 뒤, 
        post process (128,128) 로 변환하여 
        이를 흑백 이미지로 출력
    '''    
  
    # output tensor (1,1,128,128)  to output image (128,128)
    output_image = output_tensor.squeeze(0).squeeze(0)  # Remove batch dimension
    display_image(output_image)