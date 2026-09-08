"""
predict.py - Single-image inference script.

Usage
-----
  python src/predict.py --model_path outputs/best_densenet.keras \
                        --image_path path/to/xray.jpeg

Prints the predicted class (NORMAL or PNEUMONIA) and the confidence score.
"""

import argparse
import numpy as np
import tensorflow as tf
from PIL import Image

from data import IMG_SIZE, CLASS_NAMES


def predict_single(model_path: str, image_path: str) -> None:
    """Load model, preprocess a single image, and print the prediction."""
    model = tf.keras.models.load_model(model_path)

    # -- load and preprocess --------------------------------------------------
    img = Image.open(image_path).convert("RGB")      # ensure 3-channel RGB
    img = img.resize(IMG_SIZE)                        # resize to 224 x 224
    arr = np.array(img, dtype=np.float32) / 255.0    # normalise to [0, 1]
    arr = np.expand_dims(arr, axis=0)                 # add batch dim -> (1,224,224,3)

    # -- inference ------------------------------------------------------------
    prob = model.predict(arr, verbose=0)[0][0]        # scalar sigmoid output
    predicted_class = CLASS_NAMES[int(prob >= 0.5)]

    # Confidence: how far the prediction is from the 0.5 decision boundary
    confidence = float(prob) if prob >= 0.5 else float(1.0 - prob)

    # -- output ---------------------------------------------------------------
    print()
    print(f"  Image      : {image_path}")
    print(f"  Prediction : {predicted_class}")
    print(f"  Confidence : {confidence * 100:.1f}%")
    print(f"  Raw score  : {prob:.4f}  (>= 0.5 -> PNEUMONIA)")
    if predicted_class == "PNEUMONIA":
        print()
        print("  WARNING: Pneumonia detected by the model.")
        print("           Please consult a qualified radiologist for diagnosis.")
    else:
        print()
        print("  No pneumonia detected by the model.")
        print("  NOTE: A negative result does not rule out disease.")
    print()


import os
from pathlib import Path

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    
    # Default model path
    default_model = PROJECT_ROOT / "outputs" / "best_densenet.keras"
    
    # Try finding an existing sample image from the test set
    test_dir = PROJECT_ROOT / "data" / "chest_xray" / "test" / "PNEUMONIA"
    default_image = test_dir / "person100_bacteria_475.jpeg"
    if not default_image.exists() and test_dir.exists():
        found = list(test_dir.glob("*.jpeg")) + list(test_dir.glob("*.jpg"))
        if found:
            default_image = found[0]

    parser = argparse.ArgumentParser(
        description="Run inference on a single chest X-ray image."
    )
    parser.add_argument("--model_path", default=str(default_model),
                        help="Path to saved .keras model file.")
    parser.add_argument("--image_path", default=str(default_image),
                        help="Path to chest X-ray image (JPEG or PNG).")
    args = parser.parse_args()
    predict_single(args.model_path, args.image_path)
