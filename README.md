# 🩺 Pneumonia Detection from Chest X-Rays

A deep learning-based medical image classification system that detects **Pneumonia** from chest X-ray radiographs using **Transfer Learning with DenseNet121**. The project includes model training, evaluation, inference, and a production-ready Streamlit web application.

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

✅ Interactive Streamlit Web Application

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

## 📊 Results

### Test Set Performance

| Metric                         | Score      |
| ------------------------------ | ---------- |
| ROC-AUC                        | **0.9546** |
| Accuracy                       | **86.1%**  |
| Pneumonia Recall (Sensitivity) | **96.7%**  |

### ROC Performance

A ROC-AUC score of **0.9546** indicates excellent separability between pneumonia and normal chest X-rays.

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

├── app.py
├── requirements.txt
├── samples/
├── src/
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
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
| app.py      | Streamlit application                 |

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

## 🌐 Streamlit Web Application

Launch the web interface:

```bash
streamlit run app.py
```

Features:

* Drag-and-drop image upload
* Real-time prediction
* Confidence score display
* Sample X-ray testing
* User-friendly interface

---

## 🛠 Technologies Used

* Python
* TensorFlow / Keras
* DenseNet121
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Streamlit

---

## 🔮 Future Improvements

* Multi-class lung disease classification
* Grad-CAM explainability visualizations
* Model deployment on cloud platforms
* Mobile-friendly inference API
* Larger medical imaging datasets

---

## ⚠️ Disclaimer

This project is intended for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis.

---

## 👨‍💻 Author

**Mohd Ahbab**

AI/ML engineer | Deep Learning | Computer Vision

Feel free to connect and provide feedback on the project.

```

