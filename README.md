# 🧠 Brain Tumor MRI Classification System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Made with Streamlit](https://img.shields.io/badge/Made%20with-Streamlit-FF4B4B.svg)](https://streamlit.io)
[![fastai](https://img.shields.io/badge/fastai-A80084?style=flat&logo=fastai&logoColor=white)](https://docs.fast.ai/)

---

### ⚠️ Important Medical Disclaimer
This application is for **educational and demonstration purposes only**. It is **NOT** a medical diagnostic tool and should **NOT** be used for medical decision-making. Always consult qualified healthcare professionals for medical diagnosis.

---

## 1. Title and Short Description

This project develops a deep learning model to classify brain tumors from MRI scans.

* **Problem:** Brain tumor diagnosis relies on the accurate interpretation of Magnetic Resonance Imaging (MRI) scans by radiologists. This process is manual, time-consuming, and can be subject to human error. Automating the classification of tumor types (or the absence of a tumor) can provide a valuable "second opinion" for medical professionals, speeding up diagnosis and treatment planning.
* **Importance:** This system addresses the need for rapid, accessible, and reliable screening tools. By classifying MRIs into four categories—**Glioma**, **Meningioma**, **Pituitary Tumor**, and **No Tumor**—it can help prioritize cases and assist in surgical planning.
* **Results Overview:** We trained a Convolutional Neural Network (CNN) on a dataset of brain MRI images, achieving **[...e.g., 90%+ accuracy...]** on our validation set. This trained model is deployed in an interactive [Streamlit](https://streamlit.io/) web application where users can upload an MRI scan and receive an instant classification and confidence score.

## 2. Dataset Source

The model was trained on a public dataset of brain MRI scans.

* **Data Source:** **[...e.g., The dataset was sourced from the "Brain Tumor MRI Dataset" on Kaggle, which combines data from three different sources (Figshare, SARTAJ, and Br35H)...]**
* **Data Size:** **[...e.g., The full dataset contained 7,023 images. We split this into a training set (5,712 images), a validation set (1,000 images), and a test set (311 images)...]**
* **Preprocessing:** **[...e.g., The dataset was imbalanced, so we used data augmentation techniques from `fastai` (such as rotation, zoom, and brightness adjustments) to create a more robust model. All images were resized to 224x224 pixels to match the input requirements of our pre-trained network...]**

## 3. Methods

Our approach is based on transfer learning using a pre-trained Convolutional Neural Network (CNN).

### Our Approach
We used a **[...e.g., ResNet-34...]** architecture, pre-trained on the ImageNet dataset. This approach (transfer learning) is highly effective as the model's initial layers have already learned to recognize basic features like edges and textures. We then "fine-tuned" the final layers of this network on our specific medical imaging dataset.

The training was conducted using the `fastai` library, which provides high-level abstractions for deep learning tasks. We used the `vision_learner` function to combine our pre-trained architecture with a custom classification head.


> **[...Placeholder for a diagram showing your model's architecture. e.g., A simple flowchart: `Input MRI -> ResNet-34 Base -> Custom Head (Classifier) -> Output Probabilities (Glioma, Meningioma, etc.)`...]**

### Justification
This approach was chosen for a few key reasons:
1.  **Data Efficiency:** We had a relatively small dataset (**[...e.g., ~7,000 images...]**), which is not enough to train a deep CNN from scratch. Transfer learning allows us to leverage the knowledge from the massive ImageNet dataset.
2.  **Speed:** Fine-tuning is significantly faster than training a full network.
3.  **Proven Results:** Architectures like **[...e.g., ResNet, EfficientNet, VGG...]** are proven to be highly effective for various image classification tasks, including in the medical domain.

### Alternative Approaches Considered
* **Training from Scratch:** We briefly considered training a simple CNN from scratch, but initial tests showed poor convergence and low accuracy (**[...e.g., ~60%...]**) due to the limited data.
* **Feature Extraction (non-DL):** We also considered traditional machine learning methods (like SVM or Random Forest) on features extracted via algorithms (like SIFT or HOG), but these methods are generally outperformed by end-to-end deep learning on complex image tasks.

## 4. Steps to Run the Code

This project includes a Streamlit web application for demonstrating the model. To run it locally, follow these steps.

1.  **Clone the repository**
    ```sh
    git clone [https://github.com/your_username/your_project_name.git](https://github.com/your_username/your_project_name.git)
    cd your_project_name
    ```

2.  **Create and activate a virtual environment** (Recommended)
    ```sh
    # On macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # On Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required packages**
    ```sh
    pip install -r requirements.txt
    ```

4.  **Download the trained model**
    *(Note: You must provide your trained model file, typically named `export.pkl` by `fastai`.)*
    ```sh
    # If using gdown (as implied by requirements.txt)
    gdown --id [YOUR_GOOGLE_DRIVE_FILE_ID] -O export.pkl
    ```

5.  **Run the Streamlit app**
    ```sh
    streamlit run app.py
    ```
    The application will open in your web browser at `http://localhost:8501`.

## 5. Experiments/Results Summary

We conducted several experiments to find the optimal model and hyperparameters.

### Performance Metrics
Our final model (**[...e.g., ResNet-34 fine-tuned for 5 epochs...]**) achieved the following performance on the hold-out test set:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **[...e.g., 90.1%...]** |
| Precision (Macro) | **[...e.g., 0.89...]** |
| Recall (Macro) | **[...e.g., 0.90...]** |
| F1-Score (Macro) | **[...e.g., 0.89...]** |

### Hyperparameter Tuning
We experimented with:
* **Learning Rate:** We used `fastai`'s learning rate finder (`lr_find()`) to select an optimal learning rate of **[...e.g., `1e-3`...]**.
* **Number of Epochs:** We found that performance peaked at **[...e.g., 5 epochs...]**. Training beyond this led to overfitting.
* **Architectures:** We compared **[...e.g., ResNet-18, ResNet-34, and VGG-16...]**. ResNet-34 provided the best balance of speed and accuracy for this task.

### Visualizations

**Confusion Matrix:**
The confusion matrix below shows the model's performance on the test set. We can see it performed well across all classes, with the most confusion occurring between **[...e.g., Glioma and Pituitary tumors...]**.


> **[...Placeholder for your confusion matrix. This is a critical result. You can generate this with `scikit-learn` or `fastai`'s `ClassificationInterpretation` class...]**

**Example Predictions:**
Here are some example predictions from the validation set.


> **[...Placeholder for a figure showing `(image, actual, predicted)`. `fastai`'s `interp.plot_top_losses()` is perfect for this...]**

## 6. Conclusion

This project successfully demonstrated that a deep learning model, built using transfer learning, can accurately classify brain tumor MRIs.

* **Key Learning:** We learned that **[...e.g., even with a limited dataset, transfer learning with a ResNet architecture can achieve high accuracy. Data augmentation was crucial to prevent overfitting...]**
* **Key Result:** Our final model achieved **[...e.g., 90.1% accuracy...]** and was successfully deployed as a real-time Streamlit application.
* **Future Work:** Future improvements could include **[...e.g., training on a larger, more diverse dataset; experimenting with segmentation models (like U-Net) to not only classify but also *outline* the tumor; or trying more advanced architectures like EfficientNet...]**

## 7. References

* **[...Dataset Link: e.g., "Brain Tumor MRI Dataset" on Kaggle (https://...) ...]**
* **[...Architecture Paper: e.g., "Deep Residual Learning for Image Recognition" (ResNet Paper) (https://arxiv.org/abs/1512.03385) ...]**
* **[...Libraries: `fastai` (https://docs.fast.ai/), `Streamlit` (https://streamlit.io/) ...]**
