# 🧠 Brain Tumor Classifier

[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen)](https://brain-tumor-classifier-ecru.vercel.app/)
[![React](https://img.shields.io/badge/React-19.2.0-blue)](https://reactjs.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime-orange)](https://onnxruntime.ai/)
[![Model Accuracy](https://img.shields.io/badge/Accuracy-98.38%25-success)](https://brain-tumor-classifier-ecru.vercel.app/)

> An AI-powered web application for brain tumor classification from MRI images using advanced Bayesian Deep Learning with MobileNetV2 architecture.

**🔗 [Try Live Demo](https://brain-tumor-classifier-ecru.vercel.app/)**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Performance Metrics](#performance-metrics)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Dataset](#dataset)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Overview

This project implements a state-of-the-art brain tumor classification system that analyzes MRI images to detect and classify four types of conditions:

- **Glioma** - Tumors arising from glial cells
- **Meningioma** - Tumors in the meninges
- **Pituitary** - Tumors in the pituitary gland
- **No Tumor** - Healthy brain tissue

The system achieves **98.38% accuracy** on external test data (3,264 images) and utilizes Bayesian Neural Networks for uncertainty quantification, making it highly reliable for educational and research purposes.

### ⚠️ Important Notice

**This tool is designed for educational and research purposes only. It should NOT be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical advice.**

---

## ✨ Features

### 🔬 Advanced ML Capabilities
- **Bayesian Neural Networks** with Monte Carlo Dropout for uncertainty estimation
- **Attention Mechanism** for enhanced feature extraction
- **Transfer Learning** using pre-trained MobileNetV2
- **5-Fold Cross-Validation** for robust model evaluation
- **Weighted Cross-Entropy Loss** to handle class imbalance

### 🌐 Web Application
- **Real-time Browser Inference** using ONNX Runtime Web
- **Responsive UI** with Bootstrap 5
- **Drag-and-drop Image Upload**
- **Confidence Scores** with visual probability distributions
- **Uncertainty Quantification** (Entropy & Standard Deviation)
- **Optimized for Mobile** and desktop devices

### 📊 Model Features
- **Single Image Prediction** with detailed probability breakdown
- **Batch Processing** support
- **Monte Carlo Sampling** for uncertainty estimation
- **Class Activation Maps** ready architecture
- **Production-ready ONNX format** for cross-platform deployment

---

## 🏗️ Model Architecture

### Bayesian MobileNetV2 with Attention

```
Input (224x224x3 MRI Image)
    ↓
MobileNetV2 Backbone (Pre-trained on ImageNet)
    ↓
Feature Enhancer (Depthwise Separable Convolutions)
    ↓
Squeeze-Excitation Attention Module
    ↓
Global Average Pooling
    ↓
Bayesian Classifier (3 BayesLinear Layers with MC Dropout)
    ├── Layer 1: 1280 → 640 (Dropout: 30%)
    ├── Layer 2: 640 → 320 (Dropout: 15%)
    └── Layer 3: 320 → 4 (Dropout: 7.5%)
    ↓
Output (4 Classes with Uncertainty)
```

### Key Components

1. **MobileNetV2 Backbone**
   - Efficient convolutional neural network
   - Pre-trained on ImageNet
   - Partial layer freezing for fine-tuning

2. **Attention Mechanism**
   - Squeeze-and-Excitation blocks
   - Channel-wise feature recalibration
   - Improved focus on relevant features

3. **Bayesian Layers**
   - Monte Carlo Dropout (training mode during inference)
   - Prior distributions: μ=0, σ=0.03
   - KL divergence regularization

4. **Feature Enhancer**
   - Depthwise separable convolutions
   - Batch normalization
   - ReLU6 activation

---

## 📈 Performance Metrics

### Cross-Validation Results (5-Fold on 7,023 images)

| Metric | Mean ± Std |
|--------|------------|
| **Accuracy** | 97.95% ± 0.21% |
| **Precision** | 97.97% ± 0.21% |
| **Recall** | 97.95% ± 0.21% |
| **F1-Score** | 97.95% ± 0.21% |
| **Loss** | 0.5318 ± 0.0051 |

### Final Model Performance

**Training Dataset:** 7,023 images (4 classes)
**Internal Test Dataset:** 1,054 images (15% hold-out from training)
**External Test Dataset:** 3,264 images (completely unseen data)

| Dataset | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| **Training** | 98.76% | - | - | - |
| **Validation** | 98.55% | - | - | - |
| **Internal Test** | **98.20%** | 98.24% | 98.20% | 98.20% |
| **External Test** | **98.38%** | 98.38% | 98.38% | 98.38% |

### Per-Class Performance (External Test)

| Class | Accuracy | Samples | Error Rate |
|-------|----------|---------|------------|
| **Glioma** | 98.06% | 926 | 1.94% |
| **Meningioma** | 97.33% | 937 | 2.67% |
| **No Tumor** | 99.60% | 500 | 0.40% |
| **Pituitary** | 99.11% | 901 | 0.89% |

### Training Improvements

- **Initial Model Accuracy:** 30.48%
- **After Fine-tuning:** 98.38%
- **Total Improvement:** +67.90%

---

## 🛠️ Technology Stack

### Frontend
- **React** 19.2.0 - UI framework
- **Bootstrap** 5.3.8 - Responsive design
- **ONNX Runtime Web** 1.23.0 - Browser-based inference
- **React Bootstrap** 2.10.10 - React components

### Backend/Training
- **PyTorch** 1.9.0+ - Deep learning framework
- **TorchVision** 0.10.0+ - Image processing
- **TorchBNN** - Bayesian Neural Networks
- **ONNX** - Model conversion and deployment
- **NumPy** - Numerical computing
- **Scikit-learn** - ML utilities and metrics

### Development Tools
- **Google Colab** - Model training (Tesla T4 GPU)
- **Vercel** - Web deployment
- **Git** - Version control

---

## 📁 Project Structure

```
brain-tumor-classifier/
│
├── public/
│   ├── model/
│   │   └── brain_tumor_model.onnx      # Trained ONNX model (98.38% accuracy)
│   ├── index.html
│   ├── favicon.ico
│   └── manifest.json
│
├── src/
│   ├── App.js                          # Main React component
│   ├── App.css                         # Styling
│   ├── index.js                        # Entry point
│   └── index.css                       # Global styles
│
├── package.json                        # Dependencies
├── README.md                           # This file
└── .gitignore
```

---

## 🚀 Installation

### Prerequisites

- **Node.js** 16.x or higher
- **npm** 8.x or higher
- **Modern web browser** (Chrome, Firefox, Safari, Edge)

### Local Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/rad129ratul/brain-tumor-classifier.git
   cd brain-tumor-classifier
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start development server**
   ```bash
   npm start
   ```

4. **Open browser**
   ```
   Navigate to http://localhost:3000
   ```

### Build for Production

```bash
npm run build
```

The optimized production build will be in the `build/` directory.

---

## 💻 Usage

### Web Application

1. **Visit the live demo:** [https://brain-tumor-classifier-ecru.vercel.app/](https://brain-tumor-classifier-ecru.vercel.app/)

2. **Upload an MRI image:**
   - Click "Choose File" or drag and drop
   - Supported formats: JPG, PNG, JPEG

3. **Analyze:**
   - Click "Analyze Image"
   - Wait for processing (~1-2 seconds)

4. **Review Results:**
   - **Predicted Class** with confidence percentage
   - **Probability Distribution** for all classes
   - **Uncertainty Metrics** (entropy and standard deviation)

## 🎓 Model Training

### Training Pipeline

The model was trained using a comprehensive pipeline:

#### 1. **Data Preparation**
- **Dataset Size:** 7,023 MRI images
- **Classes:** 4 (glioma, meningioma, notumor, pituitary)
- **Augmentation:** Rotation, flipping, color jitter, random erasing
- **Normalization:** ImageNet statistics

#### 2. **Training Configuration**
- **Hardware:** Tesla T4 GPU (15.83 GB)
- **Batch Size:** 64
- **Epochs:** 30
- **Optimizer:** AdamW with differential learning rates
  - Backbone: 3e-5
  - Feature Enhancer: 8e-4
  - Attention: 8e-4
  - Classifier: 1.5e-3
- **Scheduler:** OneCycleLR
- **Loss Function:** Weighted Cross-Entropy + KL Divergence
- **Label Smoothing:** 0.15

#### 3. **Training Strategy**
- **5-Fold Stratified Cross-Validation**
- **Weighted Random Sampling** for class balance
- **Early Stopping** with validation monitoring
- **Gradient Clipping** (max norm: 2.0)
- **Mixed Precision Training** (FP16)

#### 4. **Data Split**
- Training: 72.3% (5,073 images)
- Validation: 12.8% (896 images)
- Test: 15.0% (1,054 images)

### Training Results Visualization

The training process achieved:
- **Initial accuracy:** ~40% (Epoch 1)
- **Final training accuracy:** 98.74%
- **Best validation accuracy:** 98.55%
- **Convergence:** ~15-20 epochs

---

## 📊 Dataset

### Original Dataset: brain_tumor_classification7k
- **Total Images:** 7,023
- **Classes:** 4
- **Distribution:**
  - Glioma: 1,621 images (23.1%)
  - Meningioma: 1,645 images (23.4%)
  - No Tumor: 2,000 images (28.5%)
  - Pituitary: 1,757 images (25.0%)

### External Validation: brain_tumor_classification3k
- **Total Images:** 3,264
- **Purpose:** Independent testing
- **Distribution:**
  - Glioma: 926 images
  - Meningioma: 937 images
  - No Tumor: 500 images
  - Pituitary: 901 images

## 🔌 API Reference

### ONNX Model Interface

**Input:**
- **Name:** `input`
- **Shape:** `[1, 3, 224, 224]`
- **Type:** `float32`
- **Preprocessing:** Resize to 224×224, normalize with ImageNet stats

**Output:**
- **Name:** `output`
- **Shape:** `[1, 4]`
- **Type:** `float32`
- **Postprocessing:** Apply softmax for probabilities

## 🙏 Acknowledgments

- **Dataset:** Brain tumor classification datasets from public medical imaging repositories
- **Base Architecture:** MobileNetV2 by Google Research
- **Bayesian Layers:** TorchBNN library
- **ONNX Runtime:** Microsoft ONNX Runtime team
- **Training Infrastructure:** Google Colab (Tesla T4 GPU)
- **Deployment:** Vercel platform

## 📧 Contact

**Project Maintainer:** Shaikh Radwan Ahmed Ratul
- GitHub: [rad129ratul](https://github.com/rad129ratul)
- Email: your.email@example.com

**Project Link:** [https://github.com/rad129ratul/brain-tumor-classifier](https://github.com/rad129ratul/brain-tumor-classifier)

**Live Demo:** [https://brain-tumor-classifier-ecru.vercel.app/](https://brain-tumor-classifier-ecru.vercel.app/)

---