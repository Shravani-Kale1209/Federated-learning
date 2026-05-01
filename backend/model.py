"""
backend/model.py
------------
Builds the EfficientNetB0 transfer-learning model.

Two-phase approach
  Phase 1 : base frozen  → train only the head
  Phase 2 : partial unfreeze → fine-tune the top layers
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from config import INPUT_SHAPE, NUM_CLASSES, LEARNING_RATE, FINE_TUNE_LR


# ─── Build ────────────────────────────────────────────────────────────────────

def build_model(fine_tune: bool = False, num_unfrozen_layers: int = 30) -> tf.keras.Model:
    """
    Parameters
    ----------
    fine_tune           : if True, unfreeze `num_unfrozen_layers` top layers
    num_unfrozen_layers : how many layers from the top of the base to unfreeze

    Returns
    -------
    Compiled tf.keras.Model
    """
    # ── Base ──────────────────────────────────────────────────────────────────
    base = tf.keras.applications.EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=INPUT_SHAPE,
    )

    if not fine_tune:
        base.trainable = False
    else:
        # Freeze all then selectively unfreeze the top N layers
        base.trainable = True
        for layer in base.layers[:-num_unfrozen_layers]:
            layer.trainable = False

    # ── Head ──────────────────────────────────────────────────────────────────
    inputs = tf.keras.Input(shape=INPUT_SHAPE, name="input_mri")
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense_1",
    )(x)
    x = layers.Dropout(0.4, name="dropout_1")(x)
    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense_2",
    )(x)
    x = layers.Dropout(0.3, name="dropout_2")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs, outputs, name="BrainTumorClassifier")

    # ── Compile ───────────────────────────────────────────────────────────────
    lr = FINE_TUNE_LR if fine_tune else LEARNING_RATE
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def print_summary(model: tf.keras.Model):
    model.summary()
    trainable     = sum(tf.size(v).numpy() for v in model.trainable_variables)
    non_trainable = sum(tf.size(v).numpy() for v in model.non_trainable_variables)
    print(f"\nTrainable params   : {trainable:,}")
    print(f"Non-trainable params: {non_trainable:,}")
