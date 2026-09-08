# Pneumonia Detection CNN

A deep-learning pipeline that classifies chest X-ray images as **NORMAL** or **PNEUMONIA** using TensorFlow/Keras.

Two model architectures are provided:
- **Custom CNN** — lightweight baseline, trained from scratch.
- **DenseNet121** — transfer learning from ImageNet weights (recommended).

---

## Dataset

**Source:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) by Paul Mooney on Kaggle.

### Download steps

1. Install the Kaggle CLI: `pip install kaggle`
2. Place your `kaggle.json` API token in `~/.kaggle/kaggle.json`
3. Run:
   ```bash
   kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
   unzip chest-xray-pneumonia.zip -d data/
   ```

The extracted structure should look like:
```
data/
  chest_xray/
    train/
      NORMAL/
      PNEUMONIA/
    val/          ← only 16 images; we ignore this and use our own split
      NORMAL/
      PNEUMONIA/
    test/
      NORMAL/
      PNEUMONIA/
```

---

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Training

```bash
# Transfer learning with DenseNet121 (recommended)
python src/train.py --data_dir data/chest_xray --model_type densenet --epochs 30

# Custom CNN baseline
python src/train.py --data_dir data/chest_xray --model_type cnn --epochs 30
```

Saved outputs in `outputs/`:
- `best_densenet.keras` — best model checkpoint (lowest val_loss)
- `training_curves_phase1_densenet.png` — loss/accuracy curves

---

## Evaluation

```bash
python src/evaluate.py \
  --data_dir data/chest_xray \
  --model_path outputs/best_densenet.keras
```

Reports precision, recall, F1, ROC-AUC, false negatives, confusion matrix, and ROC curve.

---

## Inference (single image)

```bash
python src/predict.py \
  --model_path outputs/best_densenet.keras \
  --image_path path/to/xray.jpeg
```

---

## Limitations & Honest Caveats

> **This is a research prototype. It is NOT a cleared medical device and must NOT be used for clinical decision-making.**

| Limitation | Details |
|---|---|
| **Single-hospital dataset** | All images come from Guangzhou Women and Children's Medical Center. The model may not generalise to images from other hospitals, scanner vendors, or patient demographics. |
| **Known label-quality issues** | The community has documented annotation inconsistencies in this dataset (see Kaggle discussion threads). Some "NORMAL" images may contain pathology. |
| **No external validation** | Performance is reported on the held-out test set from the same distribution as training. Real-world performance will likely differ. |
| **Binary only** | This model only distinguishes NORMAL vs. PNEUMONIA. It cannot differentiate bacterial vs. viral pneumonia, or identify other pathologies. |
| **Not prospectively validated** | No clinical trial or prospective study has been conducted with this implementation. |

---

## Project Structure

```
pneumonia_cnn/
├── .gitignore
├── README.md
├── requirements.txt
└── src/
    ├── data.py       # Data loading, augmentation, preprocessing
    ├── model.py      # CNN and DenseNet121 model definitions
    ├── train.py      # Training script with callbacks
    ├── evaluate.py   # Evaluation: precision, recall, F1, AUC, confusion matrix
    └── predict.py    # Single-image inference
```
