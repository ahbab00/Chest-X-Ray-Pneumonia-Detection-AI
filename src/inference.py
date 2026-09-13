"""Shared, framework-independent helpers for model inference."""

from __future__ import annotations

import numpy as np
from PIL import Image

IMG_SIZE = (224, 224)
DEFAULT_THRESHOLD = 0.50
CLASS_NAMES = ("NORMAL", "PNEUMONIA")


def prepare_image(image: Image.Image) -> np.ndarray:
    """Convert a PIL image into the normalized batch expected by the model."""
    resized = image.convert("RGB").resize(IMG_SIZE)
    pixels = np.asarray(resized, dtype=np.float32) / 255.0
    return np.expand_dims(pixels, axis=0)


def classify_probability(probability: float, threshold: float = DEFAULT_THRESHOLD) -> tuple[str, float]:
    """Return the class name and confidence for a model probability."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Model probability must be between 0 and 1.")
    label = CLASS_NAMES[int(probability >= threshold)]
    confidence = probability if label == "PNEUMONIA" else 1.0 - probability
    return label, confidence
