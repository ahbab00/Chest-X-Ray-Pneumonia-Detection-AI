import argparse
import os
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf

from data import IMG_SIZE, CLASS_NAMES


def predict_single(model_path: str, image_path: str):
    model = tf.keras.models.load_model(model_path)

    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    prob = float(model.predict(arr, verbose=0)[0][0])
    predicted_class = CLASS_NAMES[int(prob >= 0.5)]
    confidence = prob if prob >= 0.5 else (1.0 - prob)

    print(f"\nImage: {image_path}")
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence * 100:.1f}%")
    print(f"Score: {prob:.4f}\n")


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    default_model = PROJECT_ROOT / "outputs" / "best_densenet.keras"

    test_dir = PROJECT_ROOT / "data" / "chest_xray" / "test" / "PNEUMONIA"
    default_image = test_dir / "person100_bacteria_475.jpeg"
    if not default_image.exists() and test_dir.exists():
        found = list(test_dir.glob("*.jpeg")) + list(test_dir.glob("*.jpg"))
        if found:
            default_image = found[0]

    parser = argparse.ArgumentParser(description="Inference on chest X-ray image")
    parser.add_argument("--model_path", default=str(default_model))
    parser.add_argument("--image_path", default=str(default_image))
    args = parser.parse_args()
    predict_single(args.model_path, args.image_path)

