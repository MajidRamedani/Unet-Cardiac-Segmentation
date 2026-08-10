import tensorflow as tf
import matplotlib.pyplot as plt
from Read_data import read_data,show_img
from Preprocessor import ImageProcessor
from Network import UNet
from sklearn.model_selection import train_test_split
import random
from tensorflow.keras.utils import plot_model
import numpy as np

def Network_Train (model, X_train, X_test, y_train, y_test, minimums , maximums, learning_rate, savemode = True):
    
    # Define optimizer and compile the model
    Optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=Optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    # Plot model architecture and save as PNG
    tf.keras.utils.plot_model(model, ".../model{}{}{}.png")

    # Define callbacks for early stopping and TensorBoard logging
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=4, monitor='val_loss'),
                  tf.keras.callbacks.TensorBoard(log_dir='logs')]
    
    # Train the model
    model.fit(X_train, y_train, validation_data=(X_test,y_test), batch_size=32, epochs=100, callbacks=[callbacks])
    
    # Retrieve training history
    loss = model.history.history['loss']
    val_loss = model.history.history['val_loss']
    training_accuracy = model.history.history['accuracy']
    validation_accuracy = model.history.history['val_accuracy']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
    
    # Plot training and validation loss
    ax1.plot(loss, 'r', label='Training loss')
    ax1.plot(val_loss, 'bo', label='Validation loss')
    ax1.set(title='Training and Validation Loss', xlabel='Epoch', ylabel='Loss Value', ylim=[0, 1.1])
    ax1.legend()
    
    # Plot training and validation accuracy
    plt.figure()
    ax2.plot(training_accuracy, 'r', label='Training Accuracy')
    ax2.plot(validation_accuracy, 'b', label='Validation Accuracy')
    ax2.set(title='Training and Validation Accuracy', xlabel='Epoch', ylabel='Accuracy Value', ylim=[0, 1.1])
    ax2.legend()

    plt.show()
    
    # Save the model if savemode is True
    if savemode:
        model.save('.../s{}e{}lr{}.h5'.format(minimums,maximums,learning_rate))
    
    
############Data############

# Define directories and image processing parameters
train_dir = 'train/'
test_dir = 'test/'
width = 160
height = 160
channel = 1
num_classes = 4

# Initialize ImageProcessor for training and testing data
image_processor = ImageProcessor('train/', 'test/', width = 160, height = 160, channel = 1, num_classes = 4)

# Read and preprocess training data using ImageProcessor
(raw_train_data, labeled_train_data) = read_data (train_dir, width, height, channel, num_classes, image_processor)
# Read and preprocess testing data using ImageProcessor
(raw_test_data, labeled_test_data) = read_data (test_dir, width, height, channel, num_classes, image_processor)
show_img(raw_train_data,labeled_train_data)

# Split the preprocessed training data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(raw_train_data, labeled_train_data, test_size=0.25, random_state=42)
show_img(X_train,y_train)

############Run Model############

# Define lists of parameters for model training
mini_list = [16,32,64]
maxi_list = [256,512,1024]
learning_list = [0.01,0.001,0.0001,0.00001]

# Iterate over combinations of mini, maxi, and learning rates
for i in mini_list:
    for j in maxi_list:
        for z in learning_list:
            # Create a UNet model with specified parameters
            unet_model = UNet((height, width, channel), num_classes, i , j)
            model = unet_model.get_model()            
            # Train the model using Network_Train function
            Network_Train (model, X_train, X_test, y_train, y_test, i, j, z, savemode = True) 
        
############Train Model############

def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input image', 'True mask', 'Predicted mask']

  # Plot each element in display_list as a subplot
  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()   
  
def prediction (model,test_img,test_GT):
    for i in range(len(test_img)):

        sample_image = test_img[i]
        sample_mask = test_GT[i]
        
        # Convert ground truth mask to visual encoding format
        sample_mask_Encoding = np.expand_dims(np.argmax(sample_mask, axis=2).astype(np.uint8), axis=-1)
        # Predict mask using the model
        prediction = model.predict(sample_image[tf.newaxis, ...])[0]
        predicted_mask = (prediction > 0.5).astype(np.uint8)
        # Convert predicted mask to visual encoding format
        predicted_mask_Encoding = np.expand_dims(np.argmax(predicted_mask, axis=2).astype(np.uint8), axis=-1)
    
        # Display input image, true mask, and predicted mask using the display function
        display([sample_image, sample_mask_Encoding,predicted_mask_Encoding])
    
prediction (model,raw_test_data,labeled_test_data)
    
    