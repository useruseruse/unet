import cv2
import os

def create_masks_from_directory(source_directory, output_directory,threshold_value=30):
    # Check if output directory exists, create it if it doesn't
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Iterate over all files in the source directory
    for filename in os.listdir(source_directory):
        file_path = os.path.join(source_directory, filename)

        # Check if the file is an image
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Read the image
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            _, mask = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
            mask_file_path = os.path.join(output_directory, f"mask_{filename}")
            
            cv2.imwrite(mask_file_path, mask)
            print(f"Mask saved: {mask_file_path}")


# Example usage
source_dir = "/Users/namjihyeon/Desktop/hankooktier/unet/src/data/imgs"
all_dir = "/Users/namjihyeon/Desktop/hankooktier/unet/src/data/Footshape_Gray_Image_All"

all_mask_dir = './data/masks/all'
result_mask_dir = './data/masks/all'

create_masks_from_directory(source_dir, result_mask_dir)
create_masks_from_directory(all_dir, all_mask_dir)
