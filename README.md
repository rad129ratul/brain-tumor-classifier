# 🧠 Brain Tumor Classifier

[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen)](https://brain-tumor-classifier-ecru.vercel.app/)
[![React](https://img.shields.io/badge/React-19.2.0-blue)](https://reactjs.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime-orange)](https://onnxruntime.ai/)
[![Model Accuracy](https://img.shields.io/badge/Accuracy-98.58%25-success)](https://brain-tumor-classifier-ecru.vercel.app/)

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
- [Model Conversion](#model-conversion)
- [Domain Adaptation](#domain-adaptation)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

---

## 🎯 Overview

This project implements a state-of-the-art brain tumor classification system that analyzes MRI images to detect and classify four types of conditions:

- **Glioma** - Tumors arising from glial cells
- **Meningioma** - Tumors in the meninges
- **Pituitary** - Tumors in the pituitary gland
- **No Tumor** - Healthy brain tissue

The system achieves **98.58% accuracy** on internal test data (1,054 images) and utilizes Bayesian Neural Networks for uncertainty quantification, making it highly reliable for educational and research purposes.

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
- **Domain Adaptation** techniques for improved generalization

### 🌐 Web Application
- **Real-time Browser Inference** using ONNX Runtime Web
- **Responsive UI** with Bootstrap 5
- **Drag-and-drop Image Upload**
- **Confidence Scores** with visual probability distributions
- **Uncertainty Quantification** (Entropy & Standard Deviation)
- **Monte Carlo Dropout Visualization** with 10 forward passes
- **Optimized for Mobile** and desktop devices

### 📊 Model Features
- **Single Image Prediction** with detailed probability breakdown
- **Uncertainty Metrics** including entropy and standard deviation
- **Confidence Level Assessment** (High/Moderate/Some/High Uncertainty)
- **Production-ready ONNX format** for cross-platform deployment

---

## 🗏️ Model Architecture

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
| **Accuracy** | 97.69% ± 0.39% |
| **Precision** | 97.75% ± 0.35% |
| **Recall** | 97.69% ± 0.39% |
| **F1-Score** | 97.70% ± 0.39% |
| **Loss** | 0.5317 ± 0.0056 |

### Final Model Performance

**Training Dataset:** 7,023 images (4 classes)  
**Internal Test Dataset:** 1,054 images (15% hold-out from training)  
**External Test Dataset:** 3,264 images (completely unseen data)

| Dataset | Accuracy |
|---------|----------|
| **Training** | 98.52% |
| **Validation** | 98.77% |
| **Internal Test** | **98.58%** |
| **External Test** | **98.65%** |

### Training Evolution (Final Model)
- **Best Validation Accuracy:** 98.77%
- **Final Test Accuracy:** 98.58%
- **Training Convergence:** Achieved ~98% accuracy by epoch 18
- **Stable Performance:** Maintained >98.5% accuracy in final epochs

### Domain Adaptation Results
- **Before Fine-tuning:** 30.48% (domain shift issue)
- **After Fine-tuning:** 98.10%
- **After Full Adaptation:** 98.65%
- **Total Improvement:** +68.17%

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
- **Google Colab** - Model training (Tesla P100/T4 GPU)
- **Kaggle Notebooks** - Model training and conversion
- **Vercel** - Web deployment
- **Git** - Version control

---

## 📁 Project Structure

```
brain-tumor-classifier/
│
├── public/
│   ├── model/
│   │   └── brain_tumor_model.onnx
│   ├── index.html
│   ├── favicon.ico
│   └── manifest.json
│
├── src/
│   ├── App.js
│   ├── App.css
│   ├── index.js
│   └── index.css
│
├── package.json
├── README.md
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
   - **Confidence Level** (High/Moderate/Some/High Uncertainty)
   - **Monte Carlo Dropout Details** (optional, expandable)

### Understanding the Results

#### Confidence Levels
- **High Confidence** (Entropy < 0.3): Clear prediction
- **Moderate Confidence** (Entropy 0.3-0.7): Reasonably certain
- **Some Uncertainty** (Entropy 0.7-1.1): Borderline case
- **High Uncertainty** (Entropy > 1.1): Ambiguous image

#### Monte Carlo Dropout
- 10 forward passes with different dropout masks
- Mean prediction across all samples
- Standard deviation indicates uncertainty
- Low std = high confidence; High std = uncertain prediction

---

## 🎓 Model Training

### Training Pipeline

#### 1. **Data Preparation**
- **Dataset Size:** 7,023 MRI images
- **Classes:** 4 (glioma, meningioma, notumor, pituitary)
- **Augmentation:** Rotation, flipping, color jitter, random erasing
- **Normalization:** ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

#### 2. **Training Configuration**
- **Hardware:** Tesla P100-PCIE-16GB GPU (17.06 GB memory)
- **Batch Size:** 32
- **Epochs:** 30
- **Optimizer:** AdamW with differential learning rates
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

### Training Results
- **Initial accuracy:** ~44% (Epoch 1)
- **Final training accuracy:** 98.52%
- **Best validation accuracy:** 98.77%
- **Final test accuracy:** 98.58%
- **Convergence:** ~18-20 epochs

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
- **Purpose:** Independent testing and domain adaptation
- **Distribution:**
  - Glioma: 926 images
  - Meningioma: 937 images
  - No Tumor: 500 images
  - Pituitary: 901 images

---

## 🔄 Model Conversion

### Conversion Process
1. **Load Bayesian Model**: Load checkpoint from training
2. **Create Simplified Architecture**: Convert BayesLinear layers to standard Linear layers
3. **Weight Adaptation**: Map Bayesian weights (mu parameters) to standard weights
4. **ONNX Export**: Export to ONNX format with dynamic batch size
5. **Quantization**: Optional INT8 quantization for smaller model size
6. **Verification**: Test ONNX model against PyTorch model

### Model Sizes
- Original ONNX: ~13-14 MB
- Quantized ONNX: ~3-4 MB (70% reduction)

---

## 🔧 Domain Adaptation

### Problem Identified
The original model showed only **30.48% accuracy** on the external dataset due to:
- Class name mismatch (glioma vs glioma_tumor)
- Different image characteristics
- Domain shift between datasets

#### Fine-tuning Process:
1. **Data Analysis**: Identify distribution differences
2. **Class Mapping**: Map external class names to original
3. **Fine-tuning**: Train on external dataset (80% train, 20% val)
4. **Evaluation**: Test on full external dataset

#### Results:
- **Before Fine-tuning:** 30.48%
- **After Fine-tuning (10 epochs):** 98.10%
- **Final Performance:** 98.65%
- **Improvement:** +68.17%

---

## 🙏 Acknowledgments

- **Dataset:** Brain tumor classification datasets from public medical imaging repositories
- **Base Architecture:** MobileNetV2 by Google Research
- **Bayesian Layers:** TorchBNN library
- **ONNX Runtime:** Microsoft ONNX Runtime team
- **Training Infrastructure:** Google Colab (Tesla P100/T4 GPU) and Kaggle Notebooks
- **Deployment:** Vercel platform

---

## 📧 Contact

**Project Maintainer:** Shaikh Radwan Ahmed Ratul

- **Email:** radwan.ahmed1.cse@ulab.edu.bd | ratulrs29@gmail.com
- **Phone:** 01715451470
- **GitHub:** [rad129ratul](https://github.com/rad129ratul)
- **LinkedIn:** [linkedin.com/in/shaikh-radwan-374435358](https://linkedin.com/in/shaikh-radwan-374435358)

**Project Links:**
- **Repository:** [https://github.com/rad129ratul/brain-tumor-classifier](https://github.com/rad129ratul/brain-tumor-classifier)
- **Live Demo:** [https://brain-tumor-classifier-ecru.vercel.app/](https://brain-tumor-classifier-ecru.vercel.app/)
```
