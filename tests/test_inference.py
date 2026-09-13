from PIL import Image
import numpy as np
import pytest

from src.inference import CLASS_NAMES, classify_probability, prepare_image


def test_prepare_image_returns_normalized_rgb_batch():
    image = Image.new("L", (20, 10), color=128)

    batch = prepare_image(image)

    assert batch.shape == (1, 224, 224, 3)
    assert batch.dtype == np.float32
    assert np.allclose(batch, 128 / 255.0)


@pytest.mark.parametrize(
    ("probability", "expected_label", "expected_confidence"),
    [(0.20, "NORMAL", 0.80), (0.50, "PNEUMONIA", 0.50), (0.95, "PNEUMONIA", 0.95)],
)
def test_classify_probability(probability, expected_label, expected_confidence):
    label, confidence = classify_probability(probability)

    assert label == expected_label
    assert confidence == expected_confidence
    assert label in CLASS_NAMES


@pytest.mark.parametrize("probability", [-0.01, 1.01])
def test_classify_probability_rejects_invalid_model_output(probability):
    with pytest.raises(ValueError):
        classify_probability(probability)
