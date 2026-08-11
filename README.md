# UNet Cardiac Segmentation

## Deep Learning-Based Automated Segmentation of Cardiac 2D MRI

![Python](https://img.shields.io/badge/Python-3.10.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.1-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.2-black)
![NumPy](https://img.shields.io/badge/NumPy-1.26.3-yellow)

This repository contains the implementation of **PrimUNet**, a fully automated deep learning framework for the segmentation of cardiac structures in **2D Cardiac MRI**.

The model is based on the **U-Net architecture** and performs pixel-wise segmentation of the:

- Left ventricle (LV)
- Right ventricle (RV)
- Myocardium

---

## Publication

Further details on the research and the underlying open question are available in:

**Ramedani, M., Moussavi, A., Memhave, T. R., & Boretius, S. (2025).**  
*Deep learning-based automated segmentation of cardiac real-time MRI in non-human primates.*  
**Computers in Biology and Medicine**

**DOI:** https://doi.org/10.1016/j.compbiomed.2025.109894

---

## Abstract

Cardiovascular magnetic resonance imaging (MRI) provides valuable information for assessing cardiac structure and function. However, manual segmentation of cardiac structures is time-consuming and subject to observer variability, highlighting the need for reliable and automated segmentation approaches.

This study developed **PrimUNet**, a fully automated 2D convolutional neural network based on the **U-Net architecture** for segmenting the left ventricle, right ventricle, and myocardium.

---

## 2D U-Net Model Architecture

We implemented a **2D U-Net architecture** for automated cardiac image segmentation.

The model implementation can be found in the [`src`](src) directory.

<p align="center">
  <img src="Images/Unet.png" width="700" alt="U-Net model architecture">
</p>

---

## Results

The following figure shows representative segmentation results and a comparison between the trained U-Net model and two independent human observers.

<p align="center">
  <img src="Images/Result.png" width="700" alt="Segmentation results and comparison with human observers">
</p>

The model achieved an **average Dice score of approximately 0.90**, demonstrating segmentation performance comparable to human observers.

---

## Key Findings

- Fully automated segmentation of cardiac structures from 2D MRI
- Segmentation of the **left ventricle, right ventricle, and myocardium**
- **Average Dice score: 0.90**
- Strong agreement with manual measurements of:
  - Left ventricular volume
  - Myocardial volume
- Performance comparable to human observers

---

## Repository Structure
.
├── Images/
├── Preprocessing/
├── src/  
└── README.md
