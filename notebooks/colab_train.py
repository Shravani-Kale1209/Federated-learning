# Brain Tumor MRI Classifier – Google Colab Notebook
# ==========================================================
# Run this entire file in Colab (GPU runtime recommended).
# File > Save a copy in Drive, then Runtime > Change runtime type > GPU

# ── Cell 1: Install dependencies ──────────────────────────────────────────────
# !pip install -q tensorflow==2.15 scikit-learn seaborn

# ── Cell 2: Mount Drive & set dataset path ───────────────────────────────────
# from google.colab import drive
# drive.mount('/content/drive')
#
# DATASET_ROOT = "/content/drive/MyDrive/brain_tumor_dataset"
# (Upload the Kaggle dataset to your Drive and point here)

# ── Cell 3: Download dataset directly from Kaggle ────────────────────────────
# !pip install -q kaggle
# from google.colab import files
# files.upload()   # upload your kaggle.json API key
# !mkdir ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
# !kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset --unzip -p /content/dataset

# ══════════════════════════════════════════════════════════════════════════════
# ACTUAL RUNNABLE CODE STARTS BELOW
# ══════════════════════════════════════════════════════════════════════════════

import os, sys, random, shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import EfficientNetB0
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing import image_dataset_from_directory

# ─── Config ───────────────────────────────────────────────────────────────────

DATASET_ROOT    = "/content/dataset"           # ← change if needed
TRAIN_DIR       = os.path.join(DATASET_ROOT, "Training")
TEST_DIR        = os.path.join(DATASET_ROOT, "Testing")
CLASSES         = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES     = 4
IMG_SIZE        = (224, 224)
BATCH_SIZE      = 32
EPOCHS_FREEZE   = 10
EPOCHS_UNFREEZE = 10
SEED            = 42

print("TF version:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))

# ─── Data loading ─────────────────────────────────────────────────────────────

train_ds_full = image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASSES,
    image_size=IMG_SIZE,
    batch_size=None,        # we'll batch after split
    shuffle=True,
    seed=SEED,
)

# Count total
total_train = train_ds_full.cardinality().numpy()
val_size    = int(total_train * 0.2)
train_size  = total_train - val_size

train_ds = (
    train_ds_full
    .take(train_size)
    .shuffle(1000, seed=SEED)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
val_ds = (
    train_ds_full
    .skip(train_size)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
test_ds = image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASSES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print(f"Train: {train_size} | Val: {val_size} | Test batches: {test_ds.cardinality()}")

# ─── Augmentation layer ───────────────────────────────────────────────────────

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomBrightness(0.1),
], name="augmentation")

# ─── Normalisation (EfficientNet expects [0, 255] — it normalises internally) ─
rescale = layers.Rescaling(scale=1.0)   # no-op; EfficientNet handles it

# ─── Sample augmented images ──────────────────────────────────────────────────

sample_images, sample_labels = next(iter(train_ds.take(1)))
fig, axes = plt.subplots(2, 5, figsize=(16, 6))
fig.suptitle("Augmentation preview", fontsize=14, fontweight="bold")
for i, ax in enumerate(axes.flat):
    if i < 5:
        ax.imshow(sample_images[i].numpy().astype("uint8"))
        ax.set_title(f"Original\n{CLASSES[sample_labels[i]]}")
    else:
        aug = data_augmentation(tf.expand_dims(sample_images[i-5], 0), training=True)
        ax.imshow(tf.squeeze(aug).numpy().astype("uint8"))
        ax.set_title("Augmented")
    ax.axis("off")
plt.tight_layout()
plt.show()

# ─── Build model ──────────────────────────────────────────────────────────────

def build_model(fine_tune=False, num_unfrozen=30):
    base = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
    )
    if not fine_tune:
        base.trainable = False
    else:
        base.trainable = True
        for layer in base.layers[:-num_unfrozen]:
            layer.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = data_augmentation(inputs, training=fine_tune)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    lr = 1e-5 if fine_tune else 1e-3
    m  = tf.keras.Model(inputs, outputs, name="BrainTumorClassifier")
    m.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return m

# ─── Phase 1: Frozen base ─────────────────────────────────────────────────────

callbacks_p1 = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "best_phase1.keras", monitor="val_accuracy",
        save_best_only=True, verbose=1
    ),
]

model = build_model(fine_tune=False)
model.summary()

print("\n──── Phase 1 Training (frozen base) ────")
history_p1 = model.fit(
    train_ds, validation_data=val_ds,
    epochs=EPOCHS_FREEZE, callbacks=callbacks_p1,
)

# ─── Phase 2: Fine-tuning ─────────────────────────────────────────────────────

callbacks_p2 = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-8, verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "best_phase2.keras", monitor="val_accuracy",
        save_best_only=True, verbose=1
    ),
]

model_ft = build_model(fine_tune=True, num_unfrozen=30)
model_ft.load_weights("best_phase1.keras")

print("\n──── Phase 2 Training (fine-tuning) ────")
history_p2 = model_ft.fit(
    train_ds, validation_data=val_ds,
    epochs=EPOCHS_UNFREEZE, callbacks=callbacks_p2,
)

# ─── Evaluation ───────────────────────────────────────────────────────────────

loss, acc = model_ft.evaluate(test_ds)
print(f"\nTest Loss: {loss:.4f}  |  Test Accuracy: {acc*100:.2f}%")

# ── Classification report
y_true, y_pred = [], []
for imgs, labels in test_ds:
    preds  = model_ft.predict(imgs, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4))

# ── Confusion matrix
cm = confusion_matrix(y_true, y_pred)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, data, title, fmt in zip(
    axes, [cm, cm.astype(float)/cm.sum(axis=1, keepdims=True)],
    ["Absolute", "Row-normalised"], ["d", ".2%"]
):
    sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
plt.suptitle("Confusion Matrix", fontsize=15, fontweight="bold")
plt.tight_layout(); plt.show()

# ── Training curves
def _merge(h1, h2, key):
    return h1.history.get(key, []) + h2.history.get(key, [])

ep = range(1, len(_merge(history_p1, history_p2, "accuracy")) + 1)
split = len(history_p1.history["accuracy"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
for ax, key, title in [
    (ax1, "accuracy", "Accuracy"),
    (ax2, "loss",     "Loss"),
]:
    ax.plot(ep, _merge(history_p1, history_p2, key),      label="Train")
    ax.plot(ep, _merge(history_p1, history_p2, f"val_{key}"), label="Val", linestyle="--")
    ax.axvline(split + 0.5, color="red", linestyle=":", label="Fine-tune start")
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
plt.suptitle("Training History", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.show()

# ─── Save final model ─────────────────────────────────────────────────────────
model_ft.save("brain_tumor_classifier_final.keras")
print("✓ Model saved as brain_tumor_classifier_final.keras")

# ── Download it to local machine:
# from google.colab import files
# files.download("brain_tumor_classifier_final.keras")
