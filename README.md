# Handwritten-Character-Recognition-Deep-Learning-Project-

This repository contains a deep learning–based system for recognizing handwritten English alphabet characters (A–Z). The project applies advanced convolutional neural networks (CNNs) and transfer learning using MobileNetV2 to classify 26 distinct characters with high accuracy.

This work demonstrates strong skills in computer vision, deep learning, hyperparameter tuning, model optimization, and pipeline development — and can be integrated into OCR systems, digital forms processing, educational applications, and assistive technologies.

# 🎯Objective 

To build a robust image-based recognition system that:

Accurately identifies handwritten alphabet characters (A–Z)

Learns discriminative visual patterns from grayscale images

Handles variations in handwriting style through augmentation

Compares multiple deep learning architectures

Produces a deployable, scalable classification model

This project showcases an end-to-end deep learning workflow suitable for real-world computer-vision use cases.

# 🧠Key Capabilities

Multi-Architecture Training: Baseline CNN, Regularized CNN, Hyperparameter-Tuned CNN, MobileNetV2 (frozen, fine-tuned, and tuned versions)

High-Accuracy Image Classification: Predicts A–Z characters from 28×28 grayscale images

Data Augmentation: Rotation, shift, zoom, and normalization for better generalization

Transfer Learning: Leverages pretrained MobileNetV2 for improved feature extraction

Model Optimization: Dropout, L2 regularization, Keras Tuner search

Evaluation Suite: Confusion matrix, classification report, learning curves

Reproducible Pipeline: Fully implemented in Jupyter Notebook / Python with TensorFlow

# 🏗️ Technical Approach

The system follows a structured computer-vision and deep-learning pipeline:

1. Data Preparation

Balanced a subset of the original A–Z dataset (400 samples per class)

Normalized pixel intensities (0–255 → 0–1)

Reshaped inputs for CNNs and MobileNetV2

Performed stratified splitting (train/val/test)

2. Exploratory Analysis

Visualized class distribution

Displayed random sample images

Inspected pixel intensity patterns

3. Model Development
A. Baseline CNN (3-layer)

Convolution + MaxPooling layers

Dense classifier

Softmax output for 26 classes

B. Regularized CNN

Dropout layers to reduce overfitting

L2 regularization

Early stopping and reduced learning rate scheduling

C. Hyperparameter-Tuned CNN

Keras Tuner search space:

Dropout rates

Dense units

Learning rate

L2 strength

Rebuilt the best model and retrained

D. MobileNetV2 (Transfer Learning)

Frozen feature extractor

Added global pooling + dense head

Fine-tuned top layers for handwriting adaptation

Hyperparameter tuning for optimal performance

4. Training and Optimization

20–60 epochs depending on the model

Adam optimizer

Batch size = 64

Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

5. Evaluation

Test set accuracy

F1-score and precision/recall for each character

Confusion matrix (raw + normalized)

Learning curves: accuracy & loss

This ensures thorough model validation and interpretability.

# 🛠️ Technologies Used

Python

TensorFlow / Keras

MobileNetV2 (ImageNet weights)

NumPy, Pandas

Matplotlib, Seaborn

Keras Tuner

Jupyter Notebook

# 📈 Results Summary

Your experiments produced:

Strong performance from the Regularized CNN

Improved generalization from Fine-Tuned MobileNetV2

# 🚀How It Works (High-Level)

Load and preprocess the dataset

Train multiple CNN models

Compare deep-learning architectures

Apply transfer learning with MobileNetV2

Fine-tune and hyper-tune for best accuracy

Evaluate and visualize model performance

Save best-performing model for deployment

Best stability and accuracy using Hyperparameter-Tuned MobileNetV2

This demonstrates successful application of classical CNNs and state-of-the-art transfer learning.
