# ML_Project_22070521012

# 🧠 Brain Tumor MRI Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Made with Streamlit](https://img.shields.io/badge/Made%20with-Streamlit-FF4B4B.svg)](https://streamlit.io)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A Streamlit web application that uses a deep learning model to classify brain MRI images into four categories: Glioma, Meningioma, Pituitary, or No Tumor.

---

### ⚠️ Important Medical Disclaimer
This application is for **educational and demonstration purposes only**. It is **NOT** a medical diagnostic tool and should **NOT** be used for medical decision-making. Always consult qualified healthcare professionals for medical diagnosis.

---

## 📸 Project Showcase

## 📋 Table of Contents

- [About The Project](#-about-the-project)
- [Built With](#-built-with)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#-usage)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

## 📖 About The Project

This project provides a user-friendly web interface for classifying brain tumor MRI scans. It is designed to demonstrate the power of deep learning models (specifically, a Convolutional Neural Network built with `fastai`) in the medical imaging field.

The model classifies an uploaded MRI image into one of four distinct categories:
* 🧬 **Glioma**
* 🧠 **Meningioma**
* 🌀 **Pituitary Tumor**
* ✅ **No Tumor**

The application not only provides a prediction but also a confidence score, a full probability breakdown, and a brief medical interpretation for educational context.

### 🛠️ Built With

This project leverages several powerful open-source libraries:

* **Core Framework:**
    * [cite_start][Streamlit](https://streamlit.io/) [cite: 1]
    * [cite_start][fastai](https://docs.fast.ai/) [cite: 1]
    * [cite_start][Torchvision](https://pytorch.org/vision/stable/index.html) [cite: 1]
* **Data & ML:**
    * [cite_start][Scikit-learn](https://scikit-learn.org/stable/) [cite: 1]
    * [cite_start][NumPy](https://numpy.org/) [cite: 1]
* **Image Handling & Plotting:**
    * [cite_start][Pillow (PIL)](https://python-pillow.org/) [cite: 1]
    * [cite_start][Matplotlib](https://matplotlib.org/) [cite: 1]
* **Utilities:**
    * [cite_start][gdown](https://github.com/wkentaro/gdown) [cite: 1] (Likely for downloading the trained model file)

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

You must have Python 3.8 (or newer) and `pip` installed on your system.

### Installation

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
    [cite_start]This will install all dependencies listed in your `requirements.txt` file[cite: 1].
    ```sh
    pip install -r requirements.txt
    ```

4.  **Download the trained model** (If applicable)
    [cite_start]*(**Note:** Your current `app.py` uses mock data. When you integrate your real model, you'll add the download step here. The `gdown` library [cite: 1] suggests you'll host it on Google Drive.)*
    ```sh
    # Example: gdown --id YOUR_MODEL_FILE_ID -O models/export.pkl
    ```

## 🏃 Usage

Once the installation is complete, you can run the application locally:

1.  **Run the Streamlit app**
    ```sh
    streamlit run app.py
    ```

2.  **Open the application**
    Streamlit will automatically open the app in your default web browser (usually at `http://localhost:8501`).

3.  **Use the app**
    * Upload a brain MRI scan (JPG, JPEG, or PNG).
    * Click the "🚀 Analyze Image" button.
    * View the prediction, confidence score, and probability breakdown.

## 🗺️ Roadmap

The current `app.py` uses simulated data (`np.random.dirichlet`) for demonstration purposes. The next steps are to integrate the fully trained model.

* [ ] **Integrate Real Model:** Replace the mock `analyze_mri_image` function with logic to load your saved `fastai`/PyTorch model (`.pkl` or `.pth` file).
* [cite_start][ ] **Model Loading:** Use `gdown` [cite: 1] or a similar tool to download the pre-trained model file during setup or on first launch.
* [cite_start][ ] **Image Preprocessing:** Add the necessary `fastai`/`torchvision` [cite: 1] transforms to prepare the uploaded image for the model.
* [ ] **UI Enhancements:** Add more detailed medical information or visualizations.

See the [open issues](https://github.com/TanishqGhodpage/ML_Project_22070521012/issues) for a full list of proposed features (and known issues).

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📄 License

Distributed under the **MIT License**. See `LICENSE.md` for more information. (You should add a `LICENSE.md` file with the MIT license text to your project).

## 📧 Contact

Your Name - `Tanishq Ghodpage`

Project Link: `(https://github.com/TanishqGhodpage/ML_Project_22070521012)`
