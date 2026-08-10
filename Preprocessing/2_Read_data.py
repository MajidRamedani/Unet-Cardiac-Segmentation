import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bar
from skimage.io import imread
import nibabel as nib
from tensorflow.keras.utils import to_categorical  # For one-hot encoding
import random

def read_data(Main_folder, width, height, channel, num_classes, Processor):
    # Get list of folders (each containing images and masks)
    data_list = next(os.walk(Main_folder))[1]
    
    # Initialize arrays to store train data and labeled data
    raw_data = np.zeros((len(data_list), height, width, channel), dtype=np.uint8)
    labeled_data = np.zeros((len(data_list), height, width, num_classes), dtype=np.uint8)

    print('Start Reading {}Data...'.format(Main_folder))
    
    # Loop through each folder in Main_folder
    for n, folder_name in tqdm(enumerate(data_list), total=len(data_list)):
        path = os.path.join(Main_folder, folder_name)
        tmp_mask = np.zeros((height, width, channel), dtype=np.uint8)
        
        for file in os.listdir(path):
            if file.endswith(".png"): # Process train images
                img = imread(os.path.join(path, file))
                
                # Resize and expand image using Processor methods
                resized_image, coord_x, coord_y = Processor.resize_images(img)
                resized_image[coord_y:coord_y + img.shape[0], coord_x:coord_x + img.shape[1]] = img
                
                # Store resized images in raw_data     
                raw_data[n] = Processor.expand_images(resized_image)
        
            elif file.endswith(".nii.gz"): # Process NIfTI files (segmented)
                mask_init = nib.load(os.path.join(path, file))
                mask_data = np.transpose(np.squeeze(mask_init.get_fdata()), (1, 0))
                
                # Resize and expand mask using Processor methods
                resized_mask, coord_x, coord_y = Processor.resize_images(mask_data)
                resized_mask[coord_y:coord_y + mask_data.shape[0], coord_x:coord_x + mask_data.shape[1]] = mask_data
                tmp_mask = np.maximum(tmp_mask, Processor.expand_images(resized_mask))
                
        # Convert mask to categorical and store in labeled_data        
        labeled_data[n] = to_categorical(tmp_mask, num_classes)

    return (raw_data, labeled_data)

def show_img(X_train,y_train):
    
    # Choose a random image index
    image_x = random.randint(0, len(X_train))
    # Create a figure with 5 subplots
    fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(1,5,figsize=(10,8))
    # List of images and their corresponding titles
    images = [X_train[image_x], y_train[image_x, :, :, 0], y_train[image_x, :, :, 1], y_train[image_x, :, :, 2], y_train[image_x, :, :, 3]]
    titles = ["Input Image", "Background" ,"Left ventricle", "Myocardium", "Right ventricle"]
    
    # Plot each image with its title
    for ax, img, title in zip([ax1, ax2, ax3, ax4, ax5], images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    plt.show()
    
    