"""
data.py - Data loading, augmentation, and preprocessing pipeline.

The Kaggle chest-xray-pneumonia dataset has three top-level splits:
  chest_xray/
      train/   NORMAL/  PNEUMONIA/
      val/     NORMAL/  PNEUMONIA/   (only 16 images -- too small to rely on)
      test/    NORMAL/  PNEUMONIA/

We ignore the provided val/ split and carve our own 20% validation slice out
of train/ using a deterministic random seed so results are reproducible.
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

# -- constants ----------------------------------------------------------------
IMG_SIZE   = (224, 224)   # spatial resolution fed to every model
BATCH_SIZE = 32
AUTOTUNE   = tf.data.AUTOTUNE
SEED       = 42           # kept constant for reproducibility

# Class index mapping produced by image_dataset_from_directory (alphabetical)
# NORMAL -> 0 , PNEUMONIA -> 1
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def build_augmentation_layer() -> tf.keras.Sequential:
    """
    Returns a Keras Sequential model that applies stochastic augmentations.
    These layers are active only during training (training=True) and are
    identity operations at inference time.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),   # mirror left <-> right
        tf.keras.layers.RandomRotation(0.05),       # +/- 18 degrees
        tf.keras.layers.RandomZoom(0.10),           # +/- 10% zoom
        tf.keras.layers.RandomContrast(0.10),       # +/- 10% contrast
    ], name="augmentation")


def load_datasets(data_dir: str, val_split: float = 0.20):
    """
    Load train / validation / test tf.data.Dataset objects.

    Parameters
    ----------
    data_dir   : Root directory containing train/ and test/ sub-folders.
    val_split  : Fraction of training images to reserve for validation.

    Returns
    -------
    train_ds, val_ds, test_ds  : Batched, prefetched tf.data.Dataset objects.
    class_weights              : Dict {class_index: weight} for loss weighting.
    """
    train_root = os.path.join(data_dir, "train")
    test_root  = os.path.join(data_dir, "test")

    # -- training split (80%) -------------------------------------------------
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_root,
        validation_split=val_split,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",   # single float 0.0 / 1.0
        shuffle=True,
    )

    # -- validation split (20%) -----------------------------------------------
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_root,
        validation_split=val_split,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False,         # keep order stable for metrics
    )

    # -- test set -------------------------------------------------------------
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_root,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False,
    )

    # -- class weights (to handle ~3:1 PNEUMONIA:NORMAL imbalance) ------------
    # Collect raw integer labels from the training portion to compute weights.
    labels = np.concatenate([y.numpy() for _, y in train_ds], axis=0).ravel().astype(int)
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    class_weights = dict(zip(classes.tolist(), weights.tolist()))

    # -- augmentation + normalisation -----------------------------------------
    augment = build_augmentation_layer()

    def preprocess_train(images, labels):
        images = augment(images, training=True)   # augment only during training
        images = images / 255.0                   # scale pixel values to [0, 1]
        return images, labels

    def preprocess_eval(images, labels):
        images = images / 255.0
        return images, labels

    train_ds = (train_ds
                .map(preprocess_train, num_parallel_calls=AUTOTUNE)
                .prefetch(AUTOTUNE))

    val_ds   = (val_ds
                .map(preprocess_eval, num_parallel_calls=AUTOTUNE)
                .prefetch(AUTOTUNE))

    test_ds  = (test_ds
                .map(preprocess_eval, num_parallel_calls=AUTOTUNE)
                .prefetch(AUTOTUNE))

    return train_ds, val_ds, test_ds, class_weights


def get_test_labels(test_ds: tf.data.Dataset) -> np.ndarray:
    """Extract all ground-truth labels from a dataset (used in evaluation)."""
    return np.concatenate([y.numpy() for _, y in test_ds], axis=0).ravel()
