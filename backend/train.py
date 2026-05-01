"""
backend/train.py
------------
Full two-phase training pipeline:
  Phase 1 – Frozen base, train head only
  Phase 2 – Partial unfreeze, fine-tune top layers with lower LR
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import tensorflow as tf
import numpy as np

from config import (
    CHECKPOINTS, LOGS_DIR, MODEL_NAME,
    EPOCHS_FREEZE, EPOCHS_UNFREEZE, BATCH_SIZE,
)
from backend.data_loader import get_datasets
from backend.model import build_model, print_summary


# ─── Callbacks ────────────────────────────────────────────────────────────────

def make_callbacks(phase: str) -> list:
    ckpt_path = os.path.join(CHECKPOINTS, f"{MODEL_NAME}_{phase}.keras")
    log_dir   = os.path.join(LOGS_DIR, f"{MODEL_NAME}_{phase}")

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
        ),
    ]


# ─── Training logic ───────────────────────────────────────────────────────────

def train():
    print("\n" + "=" * 60)
    print("   Brain Tumor MRI Classifier - Training")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Loading datasets ...")
    train_ds, val_ds, test_ds, class_weights, info = get_datasets()

    print(f"  Train : {info['train_size']:,} images")
    print(f"  Val   : {info['val_size']:,} images")
    print(f"  Test  : {info['test_size']:,} images")
    print(f"  Class distribution (train): {info['class_counts_train']}")

    # ── Phase 1: Frozen base ──────────────────────────────────────────────────
    print("\n[2/5] Phase 1 - Training head with frozen base ...")
    model = build_model(fine_tune=False)
    print_summary(model)

    history_p1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FREEZE,
        class_weight=class_weights,
        callbacks=make_callbacks("phase1"),
        verbose=1,
    )

    # ── Phase 2: Fine-tuning ──────────────────────────────────────────────────
    print("\n[3/5] Phase 2 - Fine-tuning top layers ...")
    model_ft = build_model(fine_tune=True, num_unfrozen_layers=30)

    # Load best weights from phase 1
    best_p1 = os.path.join(CHECKPOINTS, f"{MODEL_NAME}_phase1.keras")
    if os.path.exists(best_p1):
        model_ft.load_weights(best_p1)
        print(f"  Loaded best Phase-1 weights from: {best_p1}")

    history_p2 = model_ft.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_UNFREEZE,
        class_weight=class_weights,
        callbacks=make_callbacks("phase2"),
        verbose=1,
    )

    # ── Evaluate on test set ──────────────────────────────────────────────────
    print("\n[4/5] Evaluating on test set ...")
    loss, acc = model_ft.evaluate(test_ds, verbose=1)
    print(f"\n  Test Loss     : {loss:.4f}")
    print(f"  Test Accuracy : {acc * 100:.2f}%")

    # ── Save final model ──────────────────────────────────────────────────────
    final_path = os.path.join(CHECKPOINTS, f"{MODEL_NAME}_final.keras")
    model_ft.save(final_path)
    print(f"\n[5/5] Final model saved -> {final_path}")

    return model_ft, history_p1, history_p2


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Use GPU if available
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU(s) detected: {[g.name for g in gpus]}")
    else:
        print("No GPU detected - training on CPU (will be slow)")

    train()
