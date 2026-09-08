import os
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
SEED = 42
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def build_augmentation_layer() -> tf.keras.Sequential:
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomContrast(0.10),
    ], name="augmentation")


def load_datasets(data_dir: str, val_split: float = 0.20):
    train_root = os.path.join(data_dir, "train")
    test_root = os.path.join(data_dir, "test")

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
        shuffle=False,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_root,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False,
    )

    labels = np.concatenate([y.numpy() for _, y in train_ds], axis=0).ravel().astype(int)
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    class_weights = dict(zip(classes.tolist(), weights.tolist()))

    augment = build_augmentation_layer()

    def preprocess_train(images, labels):
        images = augment(images, training=True)
        images = images / 255.0
        return images, labels

    def preprocess_eval(images, labels):
        images = images / 255.0
        return images, labels

    train_ds = train_ds.map(preprocess_train, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_ds = val_ds.map(preprocess_eval, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    test_ds = test_ds.map(preprocess_eval, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_weights


def get_test_labels(test_ds: tf.data.Dataset) -> np.ndarray:
    return np.concatenate([y.numpy() for _, y in test_ds], axis=0).ravel()
