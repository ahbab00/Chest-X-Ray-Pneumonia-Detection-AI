import argparse
import json
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)

try:  # Supports both `python src/evaluate.py` and package imports.
    from .data import get_test_labels, load_datasets
    from .inference import CLASS_NAMES, DEFAULT_THRESHOLD
except ImportError:  # pragma: no cover - exercised by command-line scripts
    from data import get_test_labels, load_datasets
    from inference import CLASS_NAMES, DEFAULT_THRESHOLD

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = str(PROJECT_ROOT / "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def evaluate(data_dir: str, model_path: str, threshold: float = DEFAULT_THRESHOLD):
    if not 0 < threshold < 1:
        raise ValueError("threshold must be greater than 0 and less than 1.")
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    _, _, test_ds, _ = load_datasets(data_dir)
    model = tf.keras.models.load_model(model_path)
    print(f"Loaded model: {model_path}")

    y_true = get_test_labels(test_ds)
    y_prob = model.predict(test_ds, verbose=1).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS".center(60))
    print("=" * 60)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))

    auc = roc_auc_score(y_true, y_prob)
    print(f"ROC-AUC: {auc:.4f}")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fn = cm[1][0]
    total_pneumonia = cm[1].sum()
    fn_rate = fn / total_pneumonia * 100 if total_pneumonia else 0.0
    print(f"\nFalse Negatives: {fn} / {total_pneumonia} ({fn_rate:.1f}%)")
    print("=" * 60 + "\n")

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved: {cm_path}")

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(OUTPUT_DIR, "roc_curve.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"Saved: {roc_path}")

    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "threshold": threshold,
        "test_cases": int(len(y_true)),
        "roc_auc": float(auc),
        "accuracy": float(report["accuracy"]),
        "pneumonia_precision": float(report["PNEUMONIA"]["precision"]),
        "pneumonia_recall": float(report["PNEUMONIA"]["recall"]),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "confusion_matrix": [[int(value) for value in row] for row in cm],
        "model_path": str(Path(model_path).resolve()),
    }
    metrics_path = Path(OUTPUT_DIR) / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate pneumonia detection model")
    parser.add_argument("--data_dir", default=str(PROJECT_ROOT / "data" / "chest_xray"))
    parser.add_argument("--model_path", default=str(PROJECT_ROOT / "outputs" / "best_densenet.keras"))
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()
    evaluate(args.data_dir, args.model_path, args.threshold)

