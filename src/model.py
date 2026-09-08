import tensorflow as tf


def _compile(model: tf.keras.Model, lr: float) -> tf.keras.Model:
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


def build_custom_cnn(input_shape=(224, 224, 3), dropout_rate=0.5) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    for filters in [32, 64, 128, 256]:
        x = tf.keras.layers.Conv2D(filters, kernel_size=3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs, name="custom_cnn")
    return _compile(model, lr=1e-3)


def build_densenet_model(input_shape=(224, 224, 3), dropout_rate=0.5) -> tf.keras.Model:
    base = tf.keras.applications.DenseNet121(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.densenet.preprocess_input(inputs * 255.0)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs, name="densenet121_transfer")
    return _compile(model, lr=1e-3)


def unfreeze_top(model: tf.keras.Model, n_layers=100, new_lr=1e-5) -> tf.keras.Model:
    base = next(l for l in model.layers if isinstance(l, tf.keras.Model) and "densenet" in l.name.lower())
    base.trainable = True

    for layer in base.layers[:-n_layers]:
        layer.trainable = False

    return _compile(model, lr=new_lr)

