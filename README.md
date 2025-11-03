# 🧠 Brain Cancer MRI Classification Using FastAI & TensorFlow

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Made with FastAI](https://img.shields.io/badge/Made%20with-FastAI-A80084.svg)](https://docs.fast.ai/)
[![TensorFlow](https://img.shields.io/badge/Powered%20by-TensorFlow-FF6F00.svg)](https://www.tensorflow.org/)
[![Kaggle Dataset](https://img.shields.io/badge/Data-Kaggle-blue.svg)](https://www.kaggle.com/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset)

---

### ⚠️ Medical Disclaimer
This project is for **educational and research purposes only**.  
It is **not a certified medical diagnostic system**. Always consult qualified medical professionals for real diagnoses.

---

## 1️⃣ Project Overview

This repository demonstrates an **automated brain tumor classification system** using **FastAI** and **TensorFlow**.  
The model identifies four types of MRI scans:
- **Glioma**
- **Meningioma**
- **Pituitary Tumor**
- **No Tumor**

The goal is to assist radiologists and researchers in improving early diagnosis using deep learning techniques.

---

## 2️⃣ Dataset Source

**Dataset:** [PMRAM Bangladeshi Brain Cancer - MRI Dataset](https://www.kaggle.com/datasets/orvile/pmram-bangladeshi-brain-cancer-mri-dataset)

### Details
- **Total Samples:** ≈ 3,000 MRI images  
- **Classes:** Glioma, Meningioma, Pituitary, and No Tumor  
- **Image Format:** JPG/PNG MRI brain scans  
- **Preprocessing Steps:**
  - Resized to 224×224
  - Normalized pixel values
  - Augmented via rotation, zoom, and flipping
  - 80/20 train-validation split

The dataset is accessed programmatically via `kagglehub` and managed using Python’s `pathlib` and `os` libraries.

---

## 3️⃣ Methodology

### Overview
The project integrates **FastAI** for rapid model experimentation and **TensorFlow** for fine-tuning and evaluation.

#### 🧠 Stage 1 — FastAI Model Training
- Base architecture: `ResNet34` (pretrained on ImageNet)
- Transfer learning applied with differential learning rates
- Metrics: accuracy and error rate
- Early stopping and automatic LR finder used

#### 🔬 Stage 2 — TensorFlow Fine-Tuning
- Model architecture reimplemented in TensorFlow/Keras
- Adam optimizer with learning rate = 1e-4
- Dropout and batch normalization to prevent overfitting
- Validation monitoring and model checkpoint callbacks

---

## 4️⃣ Model Architecture

