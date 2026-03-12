# Kidney Cancer Grading from Histopathology Images

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange)
![Accuracy](https://img.shields.io/badge/Best%20Accuracy-81.42%25-green)

## Overview
This project implements and compares three deep learning models for automated kidney cancer grading from histopathology images. The models classify images into 5 grades (Grade 0–4) using transfer learning on pretrained ImageNet weights.

## Objective

The objective of this project is to develop and evaluate deep learning models for automated kidney cancer grading using histopathology images. The project applies transfer learning on pretrained convolutional neural networks to classify tissue images into different cancer grades and compares the performance of multiple architectures to determine the most effective model.


## Project Context

This project was carried out under the supervision of **Dr. Shyam**, focusing on the application of deep learning techniques for automated kidney cancer grading from histopathology images. The project involves implementing and evaluating multiple CNN architectures using transfer learning.

## Project Pipeline

The overall workflow of the project follows a standard deep learning pipeline for medical image classification.

Dataset → Preprocessing → Data Augmentation → Train/Validation/Test Split → Transfer Learning → Model Training → Evaluation → Model Comparison


## Models Implemented
| Model | Test Accuracy | Test Loss | Trainable Params |
|-------|-------------|-----------|-----------------|
| **InceptionResNetV2** | **81.42%** | **0.3717** | 920K |
| VGG19 | 77.27% | 0.5029 | 12.9M |
| ResNet50 | 55.73% | 0.9796 | 1.18M |

## Dataset
- **Name:** KMC Kidney Grading Dataset
- **Classes:** Grade 0, Grade 1, Grade 2, Grade 3, Grade 4
- **Training:** 3,432 images
- **Validation:** 503 images
- **Test:** 506 images

## Project Structure
```
kidney-cancer-grading-dl/
├── notebooks/         # Jupyter notebook with full pipeline
├── results/           # Confusion matrices and training curves
├── images/            # Sample histopathology images
├── requirements.txt   # Dependencies
└── README.md
```

## Methodology
- **Transfer Learning** — pretrained ImageNet weights, frozen base layers
- **Custom Classification Head** — Dense + BatchNorm + Dropout layers
- **Data Augmentation** — rotation, flips, zoom, shifts
- **Callbacks** — EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
- **Image Size** — 224×224 pixels
- **Optimizer** — Adam (lr=0.0001)
- **Epochs** — 30

## Results
InceptionResNetV2 achieved the best performance with **81.42% test accuracy**, outperforming VGG19 (77.27%) and ResNet50 (55.73%). The model showed stable convergence with closely aligned training and validation curves.

## How to Run
1. Clone the repo
```bash
   git clone https://github.com/1ms23et007-art/kidney-cancer-grading-dl.git
```
2. Install dependencies
```bash
   pip install -r requirements.txt
```
3. Open `notebooks/kidney_grading.ipynb` in Google Colab
4. Mount your Google Drive and update the dataset path
5. Run all cells

## Technologies
- Python 3.12
- TensorFlow / Keras
- Google Colab (T4 GPU)
- scikit-learn
- matplotlib / seaborn

## Author
**Amogh P Patil**  
B.E. Student | Machine Learning Enthusiast  
[GitHub](https://github.com/1ms23et007-art)
