# Pneumonia Detection from Chest X-Rays

Binary image classification pipeline to detect pneumonia from chest X-ray radiographs using convolutional neural networks and transfer learning (DenseNet121).

## Overview

- **Dataset:** Kaggle Chest X-Ray Images (Pneumonia)
- **Model:** DenseNet121 pretrained on ImageNet, fine-tuned on chest radiographs
- **Test Performance:**
  - **ROC-AUC:** 0.9546
  - **Pneumonia Sensitivity (Recall):** 96.7%
  - **Accuracy:** 86.1%
- **Interactive UI:** Streamlit web application with drag-and-drop upload and sample test cases

## Project Structure

```
pneumonia_cnn/
├── app.py              # Streamlit web application
├── requirements.txt    # Project dependencies
├── samples/            # Sample X-rays for quick testing
├── src/
│   ├── data.py         # Data loading and augmentation
│   ├── model.py        # Model architectures (CNN & DenseNet121)
│   ├── train.py        # Training pipeline
│   ├── evaluate.py     # Metrics, confusion matrix, ROC curve
│   └── predict.py      # Single-image inference CLI
└── outputs/
    └── best_densenet.keras # Trained model weights
```

## Quick Start

### 1. Setup Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the Web Application

```bash
streamlit run app.py
```

Opens an interactive dashboard at `http://localhost:8501`.

### 3. Training & Evaluation (CLI)

```bash
# Train DenseNet121 model
python src/train.py --data_dir data/chest_xray --model_type densenet --epochs 5

# Evaluate on test set
python src/evaluate.py --data_dir data/chest_xray --model_path outputs/best_densenet.keras

# Run prediction on a single image
python src/predict.py --image_path path/to/xray.jpeg
```

