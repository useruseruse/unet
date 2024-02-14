import cv2
import os
import torch
import numpy as np

def create_masks_from_directory(source_directory, threshold_value=10):
    output_directory = os.path.join(os.path.dirname(source_directory), 'masks')
    redirect_directory = os.path.join(os.path.dirname(source_directory), 'redirect')

    if not os.path.exists(output_directory) or not  os.path.exists(redirect_directory):
        os.makedirs(output_directory)

    # 폴더 내의 모든 파일에 대해서 iterate
    for filename in os.listdir(source_directory):
        file_path = os.path.join(source_directory, filename)

        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            _, mask = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

            mask_file_path = os.path.join(output_directory, f"mask_{filename}")
            redirect_file_path = os.path.join(redirect_directory,filename )

            cv2.imwrite(mask_file_path, mask)
            cv2.imwrite(redirect_file_path, image)
            print(f"Mask saved: {mask_file_path}")

    return output_directory

if __name__ == '__main__':
    # all_dir = "/Users/namjihyeon/Desktop/hankooktier/unet/src/data/Footshape_Gray_Image_All"
    # all_mask_dir = './data/masks'
    # redirect_dir = './data/redirect'
    
    result_dir = "../data_result/grey"
    # result_mask_dir = '../../data_result/masks'
    # redirect_dir = '../../data_result/redirect'

    create_masks_from_directory(result_dir)