"""
backend/data_loader.py
------------------
Builds augmented tf.data pipelines for train / validation / test splits.
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from config import (
    TRAIN_DIR, TEST_DIR, CLASSES, IMG_SIZE, BATCH_SIZE,
    VALIDATION_SPLIT, SEED,
    ROTATION_RANGE, ZOOM_RANGE, WIDTH_SHIFT_RANGE, HEIGHT_SHIFT_RANGE,
    HORIZONTAL_FLIP, BRIGHTNESS_RANGE,
)


# ─── File collection ──────────────────────────────────────────────────────────

def collect_files(root_dir: str) -> tuple[list, list]:
    """Return (file_paths, labels) for every image found under root_dir."""
    paths, labels = [], []
    for idx, cls in enumerate(CLASSES):
        cls_dir = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_dir):
            raise FileNotFoundError(
                f"Class folder not found: {cls_dir}\n"
                f"Make sure your dataset lives at: {root_dir}\n"
                f"Expected sub-folders: {CLASSES}"
            )
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(cls_dir, fname))
                labels.append(idx)
    return paths, labels


# ─── tf.data pipeline helpers ─────────────────────────────────────────────────

def _parse_image(path: tf.Tensor, label: tf.Tensor):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)   # Keep [0, 255] — EfficientNetB0 preprocesses internally
    return img, label


def _augment(img: tf.Tensor, label: tf.Tensor):
    """
    Pure tf.image augmentations — safe to use inside tf.data.map() / tf.function.
    Keras layer instances (RandomRotation etc.) cannot be created inside tf.function
    because they allocate tf.Variables, so we use the native tf.image ops instead.
    """
    # Horizontal flip
    if HORIZONTAL_FLIP:
        img = tf.image.random_flip_left_right(img)

    # Brightness & contrast (scaled for [0, 255] input)
    img = tf.image.random_brightness(img, max_delta=30.0)   # up to ±30 out of 255
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    img = tf.clip_by_value(img, 0.0, 255.0)

    # Random 90-degree rotation (safe for brain MRI — brain can appear at any angle)
    img = tf.image.rot90(img, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

    # Simulate zoom via random crop + resize
    h, w = IMG_SIZE
    margin = int(h * ZOOM_RANGE)
    crop_h = tf.random.uniform([], h - margin, h, dtype=tf.int32)
    crop_w = tf.random.uniform([], w - margin, w, dtype=tf.int32)
    img = tf.image.random_crop(img, size=[crop_h, crop_w, 3])
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.clip_by_value(img, 0.0, 255.0)

    return img, label


def _build_dataset(
    paths: list,
    labels: list,
    augment: bool = False,
    shuffle: bool = False,
) -> tf.data.Dataset:
    paths_t  = tf.constant(paths)
    labels_t = tf.constant(labels, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths_t, labels_t))

    if shuffle:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(_parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# ─── Public API ───────────────────────────────────────────────────────────────

def get_datasets():
    """
    Returns
    -------
    train_ds, val_ds, test_ds : tf.data.Dataset
    class_weights              : dict  {int: float}
    dataset_info               : dict  with size counts
    """
    # Train / val split
    train_paths, train_labels = collect_files(TRAIN_DIR)
    tr_paths, val_paths, tr_labels, val_labels = train_test_split(
        train_paths, train_labels,
        test_size=VALIDATION_SPLIT,
        stratify=train_labels,
        random_state=SEED,
    )

    # Test set
    test_paths, test_labels = collect_files(TEST_DIR)

    train_ds = _build_dataset(tr_paths,  tr_labels,  augment=True,  shuffle=True)
    val_ds   = _build_dataset(val_paths, val_labels, augment=False, shuffle=False)
    test_ds  = _build_dataset(test_paths, test_labels, augment=False, shuffle=False)

    # Compute class weights to handle imbalance
    total   = len(tr_labels)
    counts  = np.bincount(tr_labels, minlength=len(CLASSES))
    weights = {i: total / (len(CLASSES) * c) for i, c in enumerate(counts)}

    info = {
        "train_size": len(tr_paths),
        "val_size":   len(val_paths),
        "test_size":  len(test_paths),
        "class_counts_train": dict(zip(CLASSES, counts.tolist())),
    }

    return train_ds, val_ds, test_ds, weights, info
