# 🩺 Pneumonia Detection from Chest X-Rays

A deep learning-based research prototype that classifies chest X-ray radiographs for a **pneumonia signal** using transfer learning with DenseNet121. It includes training, evaluation, a FastAPI backend, and a responsive HTML/CSS/JavaScript frontend.

---

## 📌 Project Overview

Pneumonia is a serious lung infection that can be identified through chest radiographs. Manual diagnosis requires experienced radiologists and can be time-consuming.

This project leverages **Convolutional Neural Networks (CNNs)** and **Transfer Learning** to automatically classify chest X-ray images into:

* **Normal**
* **Pneumonia**

The model is trained on the Kaggle Chest X-Ray Pneumonia dataset and achieves strong classification performance while remaining lightweight enough for real-time inference.

---

## 🚀 Features

✅ Transfer Learning using DenseNet121

✅ Automated Pneumonia Detection

✅ Data Augmentation Pipeline

✅ Model Evaluation with ROC-AUC, Confusion Matrix, and Classification Metrics

✅ Single Image Prediction CLI

✅ Responsive HTML/CSS/JavaScript Web Interface

✅ FastAPI Prediction API

✅ Ready-to-Deploy Deep Learning Pipeline

---

## 🧠 Model Architecture

The final model uses **DenseNet121** pretrained on **ImageNet**.

### Why DenseNet121?

* Efficient parameter usage
* Strong feature propagation
* Reduced vanishing-gradient issues
* Excellent performance on medical imaging tasks

### Training Strategy

* Transfer Learning
* Fine-Tuning of pretrained layers
* Data Augmentation
* Binary Cross-Entropy Loss
* Adam Optimizer

---

## 📊 Results and Reproducibility

Metrics must be generated against the exact model and test dataset being used:

```bash
python src/evaluate.py --data_dir data/chest_xray --model_path outputs/best_densenet.keras
```

This writes `outputs/metrics.json`, `outputs/confusion_matrix.png`, and `outputs/roc_curve.png`. The web app displays metrics only when they are tied to its deployed model. Do not reuse results from a different dataset split, model, or threshold.

---

## 🗂 Dataset

**Dataset:** Kaggle Chest X-Ray Images (Pneumonia)

The dataset contains chest radiographs categorized into:

* Normal
* Pneumonia

Images are divided into:

* Training Set
* Validation Set
* Test Set

Dataset Link:

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

---

## 📁 Project Structure

```text
pneumonia_cnn/

├── backend/
│   ├── main.py
│   ├── schemas.py
│   └── services/
│       └── model_service.py
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── requirements.txt
├── samples/
├── src/
│   ├── inference.py
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── tests/
│   └── test_inference.py
└── outputs/
    └── best_densenet.keras
```

### Module Description

| File        | Purpose                               |
| ----------- | ------------------------------------- |
| data.py     | Data loading and augmentation         |
| model.py    | CNN and DenseNet architectures        |
| train.py    | Training pipeline                     |
| evaluate.py | Evaluation metrics and visualizations |
| predict.py  | Single image inference                |
| backend/    | FastAPI endpoints and model service   |
| frontend/   | Browser interface and image preview   |

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/your-username/pneumonia-cnn.git

cd pneumonia-cnn
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment

Windows:

```bash
.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

Python 3.11–3.13 is supported. Python 3.14 is currently unsuitable for this project's TensorFlow setup. Create and activate a virtual environment before installing dependencies.

### Verify the Project

```bash
python -m compileall -q backend src tests
pytest -q
```

---

## 🏋️ Training

Train the DenseNet121 model:

```bash
python src/train.py \
    --data_dir data/chest_xray \
    --model_type densenet \
    --epochs 5
```

---

## 📈 Evaluation

Evaluate the trained model:

```bash
python src/evaluate.py \
    --data_dir data/chest_xray \
    --model_path outputs/best_densenet.keras
```

Generated metrics include:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC Curve
* Confusion Matrix

---

## 🔍 Single Image Prediction

Run inference on a single chest X-ray:

```bash
python src/predict.py \
    --image_path path/to/xray.jpeg
```

Example Output:

```text
Prediction: Pneumonia
Confidence: 98.2%
```

---

## 🌐 Web Application

Launch the web interface:

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8501
```

Features:

* Drag-and-drop image upload
* Real-time prediction
* Confidence score display
* Sample X-ray testing
* User-friendly interface

By default the app listens only on `127.0.0.1` and limits uploads to 10 MB. Do not expose it to a network without authentication, TLS, privacy controls, and a security review.

---

## 🛠 Technologies Used

* Python
* TensorFlow / Keras
* DenseNet121
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* FastAPI and Uvicorn
* HTML, CSS, and JavaScript

---

## 🔮 Future Improvements

* Multi-class lung disease classification
* Grad-CAM explainability visualizations
* Model deployment on cloud platforms
* Mobile-friendly inference API
* Larger medical imaging datasets

---

## ⚠️ Disclaimer

This project is intended for education and research only. It is not a certified medical device, does not diagnose pneumonia, and must not be used as a substitute for a qualified clinician or radiologist. The model may be wrong, may not generalize to other populations or imaging devices, and must not be used with patient data without appropriate privacy and governance controls.

---

## 👨‍💻 Author

**Mohd Ahbab**

AI/ML engineer | Deep Learning | Computer Vision

Feel free to connect and provide feedback on the project.

```

