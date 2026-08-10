import numpy as np
from skimage.transform import resize
from zipfile import ZipFile


# Extract training data from the train.zip file
with ZipFile(".../train.zip","r") as train_data:
    train_data.extractall("./train")
    
# Extract testing data from the test.zip file
with ZipFile(".../test.zip","r") as test_data:
    test_data.extractall("./test")

class ImageProcessor:
    def __init__(self, train_dir, test_dir, width, height, channel, num_classes):
        
        # Initialize the ImageProcessor with directory paths and image specifications
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.width = width
        self.height = height
        self.channel = channel
        self.num_classes = num_classes

    def resize_images(self, data):    
        # Create an empty array for the resized image with the specified height and width        
        resized_image = np.zeros((self.height, self.width), dtype=data.dtype)
        # Calculate the new coordinates to center the image
        image_new_coord_x = (self.width - data.shape[1]) // 2
        image_new_coord_y = (self.height - data.shape[0]) // 2
        
        return (resized_image, image_new_coord_x, image_new_coord_y)

    def expand_images(self, data):    
        # Resize the image to the specified dimensions and expand its dimensions
        expanded_image = np.expand_dims(resize(data, (self.height, self.width), 
        mode='constant', preserve_range=True),axis=-1)   
        
        return expanded_image
 