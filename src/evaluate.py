"""
evaluate.py - Model evaluation on the held-out test set.

Usage
-----
  python src/evaluate.py --data_dir path/to/chest_xray \
                         --model_path outputs/best_densenet.keras

Outputs (all saved to outputs/)
  confusion_matrix.png   -- visual matrix of TP / TN / FP / FN
  roc_curve.png          -- ROC curve with AUC score
  metrics printed to stdout (precision, recall, F1, AUC, false negatives)
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)

from data import load_datasets, get_test_labels, CLASS_NAMES

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def evaluate(data_dir: str, model_path: str) -> None:
    # -- load data & model ----------------------------------------------------
    _, _, test_ds, _ = load_datasets(data_dir)
    model = tf.keras.models.load_model(model_path)
    print(f"[eval] Loaded model: {model_path}")

    # -- predictions ----------------------------------------------------------
    y_true = get_test_labels(test_ds)
    y_prob = model.predict(test_ds, verbose=1).ravel()  # sigmoid probabilities
    y_pred = (y_prob >= 0.5).astype(int)

    # -- metrics --------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"{'EVALUATION RESULTS':^60}")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    auc = roc_auc_score(y_true, y_prob)
    print(f"  ROC-AUC : {auc:.4f}")

    # False negatives = pneumonia cases predicted as NORMAL (the costly error)
    # In the confusion matrix: cm[actual][predicted]
    # cm[1][0] = actual PNEUMONIA predicted as NORMAL
    cm = confusion_matrix(y_true, y_pred)
    fn = cm[1][0]
    total_pneumonia = cm[1].sum()
    print(f"\n  False Negatives (missed pneumonia): {fn} / {total_pneumonia}")
    print(f"  ({fn / total_pneumonia * 100:.1f}% of all pneumonia cases missed)")
    print("=" * 60 + "\n")

    # -- confusion matrix plot ------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix (test set)")
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[eval] Confusion matrix saved -> {cm_path}")

    # -- ROC curve ------------------------------------------------------------
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curve (test set)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(OUTPUT_DIR, "roc_curve.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"[eval] ROC curve saved -> {roc_path}")


from pathlib import Path

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Evaluate a trained pneumonia model.")
    parser.add_argument("--data_dir",   default=str(PROJECT_ROOT / "data" / "chest_xray"),
                        help="Root of chest_xray/ dataset.")
    parser.add_argument("--model_path", default=str(PROJECT_ROOT / "outputs" / "best_densenet.keras"),
                        help="Path to saved .keras model file.")
    args = parser.parse_args()
    evaluate(args.data_dir, args.model_path)
