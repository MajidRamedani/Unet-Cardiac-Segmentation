Note
To train the network, all four files are required. However, running the 4_Train_Unet file automatically executes the others (except 5_Test_Model):

    1_Preprocessor
    2_Read_data
    3_Netwrok
    4_Train_Unet
    5_Test_Model
	

#######################
   1_Preprocessor.py
#######################

**Introduction

This script contains a code for processing images extracted from compressed .zip files. There should be two zipped files, one train, one test. 
Creation of folders within 'train' and 'test' has been explained in the README file of the Augmentation folder.

The script demonstrates also how to resize and expand images to fit specified dimensions, preparing them for deep learning models.

**Instruction

Step 1: Extracting Data

The script extracts training and testing data from zip files. Make sure you have train.zip and test.zip in the appropriate directory and define directory in your code properly. 
After executing this code(which occurs automatically), two new folders will appear in the same directory where the current script resides (It is recommended to keep all scripts and files in one directory).

This part is used in 4_Train_Unet.py


Step 2: Processing Images

The ImageProcessor class is designed to handle image preprocessing tasks. Initialize it with the directories containing training and testing data, and specify the desired image dimensions and other parameters.

def resize_images(data)

    	Description: Resizes the input image to the specified width and height, centering the image within the new dimensions.
    	Parameters:
        	data: The image data to be resized.
    	Returns:
        	resized_image: A numpy array representing the resized image.
        	image_new_coord_x: The x-coordinate for centering the image.
        	image_new_coord_y: The y-coordinate for centering the image.

** Resizing images is essential because we often deal with various types of images that have different dimensions. Standardizing them to a single unified dimension allows for more efficient and consistent processing.

def expand_images(data)

    	Description: Resizes the image to the specified dimensions and expands its dimensions.
    	Parameters:
        	data: The image data to be expanded.
    	Returns:
        	expanded_image: A numpy array representing the expanded image.

** The additional dimension is often necessary when working with CNNs that expect input data in a specific shape, typically (height, width, channels).

These functions are used in 2_Read_data.py

#######################
   2_Read_data.py
#######################

**Introduction

This Python script includes methods for reading and preparing image data, as well as visualizing images.

**Instruction

def read_data(Main_folder, width, height, channel, num_classes, Processor)

	Description: Reads image data from the folder we specified, processes and prepares it for learning.
	Parameters:

    		Main_folder: Path to the main directory containing train and test folders.
    		width: Width of the images after resizing.
    		height: Height of the images after resizing.
    		channel: Number of channels in the images (here is 1).
    		num_classes: Number of classes for categorical labeling (here is 4, LV,RV,Myo,Background).
    		Processor: Instance of ImageProcessor class for image resizing and expanding.

	Returns:

    		raw_data: Expanded data.
    		labeled_data: All labeled data (masks) corresponded to raw_data.

** We are putting images and their masks in two separate lists.

def show_img(X_train, y_train)

    	Description: Displays a random image from X_train (train data) and its corresponding labeled segmentation from y_train (mask of that trained data).
    	Parameters:
        	X_train: Trained data.
        	y_train: Labeled segmentation of trained data.
    	Output: Shows a plot with the input image and its segmented parts including background, left ventricle, myocardium, and right ventricle.

These functions are used in 4_Train_Unet.py

#######################
   3_Netwrok.py
#######################

**Introduction

This repository contains a creation of the UNet architecture, designed for image segmentation tasks. It consists of an encoder-decoder network with skip connections to preserve spatial information during downsampling and upsampling.

**Instruction

def __init__(self, input_shape, num_classes, mini, maxi)

	Description: It defines the initaila value.
	Parameters:
		input_shape: Shape of the input images (height, width, channels).
		num_classes: Number of classes to be segmented.
		mini, maxi: Number of initial and final kernel.

def bottleneck(self, last_input, n_filters, padding="same")

	Description: It consists of two convolutional layers with the same number of filters and activation function.
	Parameters:

		last_input: The input of the bottleneck layer which is the output of previous layer.
		n_filters (int): Number of kernels to be used in the convolutional layers.
		padding (str, optional): Padding strategy for convolutional layers. Defaults to "same".

	Returns:
    		conv_net: Output tensor after applying the bottleneck layers, and its input for upper layer.

def encoder_path(self, inputs, n_filters)

	Description: It defines an encoder path in a CNN. It consists of convolutional layers, dropout regularization, batch normalization, ReLU activation, max pooling, and skip connection.
	Parameters:
		inputs: The input tensor to the encoder path.
		n_filters (int): Number of kernels to be used in each layers.

	Returns:

    		next_layer: Output tensor after applying convolution, dropout, normalization, activation, and max pooling, which is the input of lower layer.
    		skip_connection: Skip connection tensor from the convolutional layer which transfered important data to decoder path.

def decoder_path(self, prev_layer, prev_skip_connection, n_filters)

	Description: It defines decoder path in CNN. It consists of transpose convolutional layers, concatenation of skip connections, convolutional layers, dropout regularization, batch normalization, and ReLU activation.
	Parameters:
		prev_layer: The previous layer tensor from the previous decoder layer.
		prev_skip_connection: The skip connection tensor from the corresponding encoder path layer.
		n_filters (int): Number of kernels to be used in the convolutional layers.

	Returns:

		conv: Output after applying different functions which is an input for upper layer.
		
def build_model(self)

	Description: This function constructs the Unet segmentation model based on specified parameters and uses encoder and decoder paths defined in the class.
	Parameters:
		inputs: Input tensor representing the image data.
		mini, maxi, num_classes: This parameters are obtained from def __init__

**The model architecture varies based on the values of self.mini and self.maxi.

Case self.mini == 16 and self.maxi == 256:

    Encoder Path:
        Four encoder blocks (encoder_path) with increasing filter sizes (16, 32, 64, 128).
    Bottleneck:
        One bottleneck layer (bottleneck) with 256 filters.
    Decoder Path:
        Four decoder blocks (decoder_path) with decreasing filter sizes (128, 64, 32, 16).
    Output Layer:
        Final convolutional layer (Conv2D) with self.num_classes filters and sigmoid activation.


def get_model(self)

	Description: It returns the built semantic segmentation model.

	Returns:
	
		model: The constructed TensorFlow Keras model for semantic segmentation.

These file is used in 4_Train_Unet.py

#######################
   4_Train_Unet.py
#######################

**Introduction

This script includes methods for training networks with different hyperparameters, as well as visualizing prediction.

**Instruction

Network_Train (model, X_train, X_test, y_train, y_test, minimums , maximums, learning_rate, savemode = True)

	Description: It is designed for training a neural network model
	Parameters:
		model: TensorFlow/Keras model object to be trained.
		X_train: Training data.
		X_test: Data used for validation.
		y_train: Training labels (target outputs).
		y_test: Validation labels (target outputs) for validation.
		minimums: Minimum number of filters.
		maximums: Maximum number of filters.
		learning_rate: Learning rate value.
		savemode: Boolean flag to determine whether to save the trained model (default=True).


	Steps:

    		Optimizer: Configures the Adam optimizer with the specified learning rate.
    		Model Compilation: Compiles the model using binary crossentropy loss and accuracy as the metric.
    		Model Visualization: Generates a plot of the model validation loss and accuracy and saves it as a PNG file.
    		Callbacks: Sets up callbacks for early stopping and TensorBoard logging.
    		Model Training: Trains the model on the training data for a specified number of epochs (default=100), with validation.
    		Performance Visualization: Plots the training and validation loss, as well as training and validation accuracy over epochs.
    		Model Saving: Optionally saves the trained model in h5 format if savemode is set to True.
	Output:

    		Model File: If savemode is True, saves the trained model in h5 format with a filename formatted as 's{}e{}lr{}.h5' based on minimums, maximums, and learning_rate.


From ""train_dir = 'train/'"" to ""num_classes = 4""

	Description: Define the directory of train and test folders, and define the specification of input and output.

From ""image_processor = "" to ""show_img""

	Description: The process starts with the 1_Preprocessor module, which reads and preprocesses both training and testing data. Subsequently, the data is split into training and validation datasets.

** Then it iterates through combinations of mini_list (minimum kernel), maxi_list (maximum kernel), and learning_list to train the UNet model using the Network_Train function in 3_Network.

def display(display_list)


	Description: It is designed to display images and their corresponding masks side by side.
	Parameters:
		display_list: A list containing three elements: input image, true mask, and predicted mask.

def prediction(model, test_img, test_GT)

	Description: It performs predictions using a given model on one random test images and visualizes the results using the display function.
	Parameters:
		model: Trained model used for prediction.
		test_img: List of test images for prediction.
		test_GT: List of corresponding ground truth masks for evaluation.
	Output:
		The prediction function displays the input image, true mask, and predicted mask for each test image using the display function.

#######################
   5_Test_Model.py
#######################
**Introduction

This repository includes calculating Dice score of model using testing dataset.

**Instruction

def dice(pred, true, k=1)

	Description: The `dice` function computes the Dice coefficient, a common metric used to evaluate the overlap between two binary masks or segmentation results.
	Parameters:
		pred (numpy array): Predicted mask array.
		true (numpy array): Ground truth mask array.
		k (int, optional): Class label for which Dice coefficient is computed. Defaults to 1.

	Return:
		dice (float): Dice coefficient for the specified class. It ranges from 0 (no overlap) to 1 (perfect overlap).


def prediction (model,test_img,test_label)

	Description: It performs segmentation predictions using a trained TensorFlow model on test images and computes evaluation metrics for each class.
	Parameters:
		model: Trained TensorFlow model for image segmentation.
		test_img (list of numpy arrays): List of test images.
		test_label (list of numpy arrays): List of corresponding ground truth masks.
	
	Return:
		means (list of lists of floats): Dice scores for each class across all test samples.
		tp (list of lists of ints): True positives for each class across all test samples.
		fp (list of lists of ints): False positives for each class across all test samples.
		fn (list of lists of ints): False negatives for each class across all test samples.
		tn (list of lists of ints): True negatives for each class across all test samples.


def print_metrics(category_name, tp_index, fp_index, fn_index, tn_index, means)

	Description: This script calculates and prints various evaluation of specificity, precision, recall, and dice score for different categories based on the performance of the model. These metrics are printed for three categories: "Blood", "Myo", and "Right".
	Parameters:
		category_name (str): The name of the category.
		tp_index (int): The index for true positives in the lists.
		fp_index (int): The index for false positives in the lists.
		fn_index (int): The index for false negatives in the lists.
		tn_index (int): The index for true negatives in the lists.
		means (list): A list containing the dice scores.

	Output:
		The script will print the specificity, precision, recall, and dice score