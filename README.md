# Unet-Cardiac-Segmentation
# **Deep learning-based automated segmentation of cardiac 2D-MRI**

![Python](https://img.shields.io/badge/Python-3.10.8+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.1-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.2-black)
![NumPy](https://img.shields.io/badge/Numpy-1.26.3-yellow)

**Further details on the addressed open question were published in:

Ramedani & et.al (2025) Deep learning-based automated segmentation of cardiac real-time MRI in non-human primates. Computers in Biology and Medicine 2025. DOI: 10.1016/j.compbiomed.2025.109894**  

---

## **Abstract**  
Cardiovascular magnetic resonance imaging (MRI) provides valuable information for assessing cardiac structure and function. However, manual segmentation of cardiac structures is time-consuming and subject to observer variability, highlighting the need for reliable automated approaches.

Therefore, this study develops PrimUNet, a fully automated 2D convolutional neural network based on the U-Net architecture for segmenting the left ventricle, right ventricle, and myocardium. PrimUNet achieved an average Dice score of 0.90 and demonstrated strong agreement with manual measurements of left ventricular volumes and myocardial volume.

---

## **2D UNET Model Architecture**  
We chose to implement 2D-Unet Architecture for our study. See [src/models](src) for implementation details.  
Following image illustrates the Unet model architecture that was utilized in this study.

<p align="center">
	<img src="Images/Unet.png" width="700">
</p>


## **Key Findings**  
Following is the result for the segmentation and its comparison with two independent observers using the trained Unet model.

<p align="center">
        <img src="Images/Result.png" width="700">
</p>


Our results demonstrate that the model attained an average Dice score of 0.9, comparable to human studies.

---

## **Citation**  
@article{RAMEDANI2025,
title = {Deep learning-based automated segmentation of cardiac real-time MRI in non-human primates},
journal = {Computers in Biology and Medicine},
volume = {189},
pages = {109894},
year = {2025},
issn = {0010-4825},
doi = {https://doi.org/10.1016/j.compbiomed.2025.109894},
url = {https://www.sciencedirect.com/science/article/pii/S0010482525002458},
author = {Majid Ramedani and Amir Moussavi and Tor Rasmus Memhave and Susann Boretius},
keywords = {Convolutional neural networks, Image segmentation, Real-time MRI, Cine MRI, Cardiac MRI, Non-human primates}
}
