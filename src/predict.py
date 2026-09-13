import argparse
from pathlib import Path
from PIL import Image
import tensorflow as tf

try:  # Supports both `python src/predict.py` and package imports.
    from .inference import DEFAULT_THRESHOLD, classify_probability, prepare_image
except ImportError:  # pragma: no cover - exercised by command-line scripts
    from inference import DEFAULT_THRESHOLD, classify_probability, prepare_image


def predict_single(model_path: str, image_path: str, threshold: float = DEFAULT_THRESHOLD):
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not Path(image_path).is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    model = tf.keras.models.load_model(model_path)

    with Image.open(image_path) as image:
        arr = prepare_image(image)

    prob = float(model.predict(arr, verbose=0)[0][0])
    predicted_class, confidence = classify_probability(prob, threshold)

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
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()
    predict_single(args.model_path, args.image_path, args.threshold)

