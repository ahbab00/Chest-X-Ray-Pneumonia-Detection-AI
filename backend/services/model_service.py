from functools import lru_cache
from io import BytesIO

from PIL import Image, UnidentifiedImageError
import tensorflow as tf

from backend.config import MODEL_PATH
from src.inference import DEFAULT_THRESHOLD, classify_probability, prepare_image

Image.MAX_IMAGE_PIXELS = 20_000_000


class ModelUnavailableError(RuntimeError):
    """Raised when the configured model cannot be loaded."""


class InvalidImageError(ValueError):
    """Raised when uploaded bytes are not a safe supported image."""


@lru_cache(maxsize=1)
def load_model() -> tf.keras.Model:
    if not MODEL_PATH.is_file():
        raise ModelUnavailableError(f"Model file not found: {MODEL_PATH}")
    try:
        return tf.keras.models.load_model(str(MODEL_PATH))
    except (OSError, ValueError) as error:
        raise ModelUnavailableError("The model file could not be loaded.") from error


def decode_image(content: bytes) -> Image.Image:
    try:
        with Image.open(BytesIO(content)) as image:
            image.verify()
        with Image.open(BytesIO(content)) as image:
            return image.convert("RGB").copy()
    except (Image.DecompressionBombError, OSError, UnidentifiedImageError) as error:
        raise InvalidImageError("The uploaded file is not a valid JPEG or PNG image.") from error


def predict(content: bytes, threshold: float = DEFAULT_THRESHOLD) -> dict:
    image = decode_image(content)
    batch = prepare_image(image)
    probability = float(load_model().predict(batch, verbose=0)[0][0])
    label, confidence = classify_probability(probability, threshold)
    return {
        "label": label,
        "confidence": confidence,
        "normal_probability": 1.0 - probability,
        "pneumonia_probability": probability,
        "threshold": threshold,
    }
