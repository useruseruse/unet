import torch
from PIL import Image
import matplotlib.pyplot as plt
from model.unet import UNet
from cosine import *

# Assuming UNet class is defined as provided

def postprocess_output(output_tensor):
    print(output_tensor.shape)
    output_image = output_tensor.squeeze(0).squeeze(0)  # Remove batch dimension
    print(output_image.shape)

    return output_image

def display_segmentation(image):
    plt.imshow(image, cmap='gray')
    plt.axis('off')  # Turn off axis numbers
    plt.show()

if __name__ == "__main__":
   
    model_path = "./model_checkpoint.pth"
    model = load_model(model_path)
    
    # Process the image
    image_path = "../data/redirect/19MA120425KI_1_60.jpg"
    input_image = transform_image(image_path)

    # Get the model output
    with torch.no_grad():
        output = model(input_image)
        print(output)
    # Postprocess and display the output
    output = postprocess_output(output)

    display_segmentation(output)