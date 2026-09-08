"""
model.py - Model definitions.

Two architectures are available:

1. build_custom_cnn()
   A lightweight CNN trained from scratch. Good as a sanity-check baseline
   and useful when compute resources are limited.

2. build_densenet_model()
   DenseNet121 backbone pretrained on ImageNet, with a custom classification
   head. Two-phase training is recommended:
     Phase 1 -- freeze the backbone, train only the head.
     Phase 2 -- unfreeze the last `unfreeze_layers` layers and fine-tune at a
                lower learning rate.
"""

import tensorflow as tf


# -- helper -------------------------------------------------------------------
def _compile(model: tf.keras.Model, lr: float) -> tf.keras.Model:
    """Compile with binary cross-entropy and standard classification metrics."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


# -- 1. Custom CNN ------------------------------------------------------------
def build_custom_cnn(input_shape: tuple = (224, 224, 3),
                     dropout_rate: float = 0.5) -> tf.keras.Model:
    """
    Lightweight baseline CNN.

    Architecture (Conv -> BN -> Pool repeated 4x, then Dense head):
      Block 1: 32 filters  -> MaxPool
      Block 2: 64 filters  -> MaxPool
      Block 3: 128 filters -> MaxPool
      Block 4: 256 filters -> MaxPool
      GlobalAveragePooling -> Dense(256) -> Dropout -> Dense(1, sigmoid)

    BatchNormalization after each Conv layer stabilises training.
    GlobalAveragePooling instead of Flatten reduces parameter count and
    acts as a structural regulariser.
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    for filters in [32, 64, 128, 256]:
        x = tf.keras.layers.Conv2D(
            filters, kernel_size=3, padding="same", activation="relu"
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs, name="custom_cnn")
    return _compile(model, lr=1e-3)


# -- 2. DenseNet121 transfer-learning model -----------------------------------
def build_densenet_model(input_shape: tuple = (224, 224, 3),
                         dropout_rate: float = 0.5) -> tf.keras.Model:
    """
    DenseNet121 backbone (ImageNet weights) with a custom binary-classification
    head.

    Phase 1: the backbone is fully frozen; only the head is trained.
    Call unfreeze_top(model, n) before Phase 2 to fine-tune the top-n layers.

    DenseNet121 was chosen because:
      - It was originally validated on chest X-rays (CheXNet, Rajpurkar 2017).
      - Dense connections allow feature reuse across depths, beneficial when
        training data is limited.
      - 224x224 input matches the ImageNet pre-training resolution.
    """
    base = tf.keras.applications.DenseNet121(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    base.trainable = False   # Phase 1: freeze entire backbone

    inputs = tf.keras.Input(shape=input_shape)
    # DenseNet121 expects inputs preprocessed by its own preprocess_input.
    # We normalised to [0,1] in data.py, so we scale back to [0,255] here.
    x = tf.keras.applications.densenet.preprocess_input(inputs * 255.0)
    x = base(x, training=False)   # training=False keeps BN in inference mode
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs, name="densenet121_transfer")
    return _compile(model, lr=1e-3)


def unfreeze_top(model: tf.keras.Model,
                 n_layers: int = 100,
                 new_lr: float = 1e-5) -> tf.keras.Model:
    """
    Unfreeze the last `n_layers` of the DenseNet backbone for fine-tuning.
    A very small learning rate (default 1e-5) prevents catastrophic forgetting
    of the pre-trained ImageNet features.
    """
    # The base model (DenseNet121) is stored as the layer at index 1
    # in the functional model: Input -> preprocess -> DenseNet -> GAP -> Dense
    base = next(l for l in model.layers
                if isinstance(l, tf.keras.Model) and "densenet" in l.name.lower())
    base.trainable = True

    # Re-freeze everything except the last n_layers of the backbone
    for layer in base.layers[:-n_layers]:
        layer.trainable = False

    return _compile(model, lr=new_lr)
