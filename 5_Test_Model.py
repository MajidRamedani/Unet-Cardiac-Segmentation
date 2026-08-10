import matplotlib.pyplot as plt
import tensorflow as tf
import random
import numpy as np
from Preprocessor import ImageProcessor
from tensorflow.keras.models import load_model
from Read_data import read_data,show_img

 
def dice(pred, true, k=1):

    pred = np.array(pred)
    true = np.array(true)
    
    # If true mask has no pixels of class 'k', return 0.0 to avoid division by zero
    if np.sum(true == k) == 0:
        return 0.0  

    # Calculate intersection
    intersection = np.sum(pred[true == k]) * 2.0
    
    # Calculate Dice coefficient
    dice = intersection / (np.sum(pred) + np.sum(true))
    return dice

def prediction (model,test_img,test_label):
    
    # Initialize lists to store results
    means, tp, fp, fn, tn=[],[],[],[],[]

    # Loop through each test image
    for i in range(len(test_img)):
        # Get the sample image and its corresponding mask
        sample_image = test_img[i]
        sample_mask = test_label[i]
        
        # Encode the sample mask
        sample_mask_Encoding = np.expand_dims(np.argmax(sample_mask, axis=2).astype(np.uint8), axis=-1)
        # Make prediction on the sample image
        prediction = model.predict(sample_image[tf.newaxis, ...])[0]
        # Encode the predicted mask
        predicted_mask = (prediction > 0.5).astype(np.uint8)
        predicted_mask_Encoding = np.expand_dims(np.argmax(predicted_mask, axis=2).astype(np.uint8), axis=-1)
        
        # Temporary lists to store metrics for each class
        tmp_mean, tmp_tp, tmp_fp, tmp_fn, tmp_tn=[],[],[],[],[]
        
        for j in range(1,4):
            
            # Calculate the Dice score for the current class
            dice_score = dice(predicted_mask[:, :, j], sample_mask[:, :, j], k=1)
            tmp_mean.append(dice_score)          
                
            # Calculate true positives, false positives, false negatives, and true negatives
            tmp_tp.append(np.sum(np.logical_and(sample_mask[:, :, j], predicted_mask[:, :, j])))
            tmp_fp.append(np.sum(np.logical_and(np.logical_not(sample_mask[:, :, j]), predicted_mask[:, :, j])))
            tmp_fn.append(np.sum(np.logical_and(sample_mask[:, :, j], np.logical_not(predicted_mask[:, :, j]))))
            tmp_tn.append(np.sum(np.logical_and(np.logical_not(sample_mask[:, :, j]),np.logical_not(predicted_mask[:, :, j]))))
            
        means.append(tmp_mean)
        tp.append(tmp_tp)
        fp.append(tmp_fp)
        fn.append(tmp_fn)
        tn.append(tmp_tn)
        
    return (means, tp, fp, fn, tn)
    

train_dir = 'train/'
test_dir = 'test/'
width = 160
height = 160
channel = 1
num_classes = 4

image_processor = ImageProcessor('train/', 'test/', width = 160, height = 160, channel = 1, num_classes = 4)
(raw_test_data, labeled_test_data) = read_data (test_dir, width, height, channel, num_classes, image_processor)
model = load_model(".../s64e512lr0.0001.h5")
(means, tp, fp, fn, tn) = prediction (model,raw_test_data,labeled_test_data)


def print_metrics(category_name, tp_index, fp_index, fn_index, tn_index, means):
    print(f"************{category_name}************")
    
    # Calculate the sum of true positives for the given index
    tpprime = sum([item[tp_index] for item in tp])
    # Calculate the sum of false positives for the given index
    fpprime = sum([item[fp_index] for item in fp])
    # Calculate the sum of false negatives for the given index
    fnprime = sum([item[fn_index] for item in fn])
    # Calculate the sum of true negatives for the given index
    tnprime = sum([item[tn_index] for item in tn])
    
    # Calculate specificity, precision, recall dice_score and ensuring no division by zero
    specificity = tnprime / (tnprime + fpprime) if (tnprime + fpprime) != 0 else 0
    precision = tpprime / (tpprime + fpprime) if (tpprime + fpprime) != 0 else 0
    recall = tpprime / (tpprime + fnprime) if (tpprime + fnprime) != 0 else 0
    dice_score = round(np.mean([item[tp_index] for item in means if item[tp_index] != 0]), 4)
    
    print("Specificity: {}".format(round(specificity, 4)))
    print("Precision: {}".format(round(precision, 4)))
    print("Recall: {}".format(round(recall, 4)))
    print("Dice Score: {}".format(dice_score))
    print()

print_metrics("Blood", 0, 0, 0, 0, means)
print_metrics("Myo", 1, 1, 1, 1, means)
print_metrics("Right", 2, 2, 2, 2, means)



