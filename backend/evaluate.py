"""
backend/evaluate.py
---------------
Generates:
  • Classification report (precision / recall / F1 per class)
  • Confusion matrix heatmap  →  results/confusion_matrix.png
  • Training history curves   →  results/training_history.png
  • Per-class accuracy bar    →  results/per_class_accuracy.png
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from config import CLASSES, CHECKPOINTS, RESULTS_DIR, MODEL_NAME
from backend.data_loader import get_datasets
from backend.load_compat import load_model_compat


# ─── Helpers ─────────────────────────────────────────────────────────────────

class CompatDense(tf.keras.layers.Dense):
    def __init__(self, *args, quantization_config=None, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config):
        config.pop("quantization_config", None)
        return super().from_config(config)


def load_best_model() -> tf.keras.Model:
    path = os.path.join(CHECKPOINTS, f"{MODEL_NAME}_final.keras")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}. Run train.py first.")
    print(f"Loading model from: {path}")
    return load_model_compat(path)


def get_predictions(model, dataset):
    y_true, y_pred = [], []
    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))
    return np.array(y_true), np.array(y_pred)


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)   # row-normalized

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Brain Tumor MRI - Confusion Matrix", fontsize=16, fontweight="bold")

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Absolute counts", "Row-normalized"],
        ["d", ".2%"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES,
            linewidths=0.5, ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


def plot_training_history(history_p1, history_p2):
    """Concatenate phase1 + phase2 history and plot."""
    def _concat(h1, h2, key):
        v1 = h1.history.get(key, [])
        v2 = h2.history.get(key, [])
        return v1 + v2

    acc     = _concat(history_p1, history_p2, "accuracy")
    val_acc = _concat(history_p1, history_p2, "val_accuracy")
    loss    = _concat(history_p1, history_p2, "loss")
    val_loss= _concat(history_p1, history_p2, "val_loss")
    ep      = range(1, len(acc) + 1)
    split   = len(history_p1.history["accuracy"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History", fontsize=15, fontweight="bold")

    for ax, y, yv, ylabel in [
        (ax1, acc, val_acc, "Accuracy"),
        (ax2, loss, val_loss, "Loss"),
    ]:
        ax.plot(ep, y,  label="Train", linewidth=2)
        ax.plot(ep, yv, label="Val",   linewidth=2, linestyle="--")
        ax.axvline(x=split + 0.5, color="red", linestyle=":", label="Fine-tune start")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "training_history.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


def plot_per_class_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    per_class = cm.diagonal() / cm.sum(axis=1)

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(CLASSES, per_class * 100, color=colors, edgecolor="white", linewidth=1.2)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-Class Accuracy", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, per_class):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{val*100:.1f}%",
            ha="center", va="bottom", fontweight="bold",
        )

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "per_class_accuracy.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def evaluate(history_p1=None, history_p2=None):
    print("\n" + "=" * 60)
    print("   Brain Tumor MRI Classifier - Evaluation")
    print("=" * 60)

    model = load_best_model()
    _, _, test_ds, _, _ = get_datasets()

    print("\nRunning predictions on test set ...")
    y_true, y_pred = get_predictions(model, test_ds)

    print("\n-- Classification Report -----------------------------")
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4))

    print("\n-- Generating plots ----------------------------------")
    plot_confusion_matrix(y_true, y_pred)
    plot_per_class_accuracy(y_true, y_pred)

    if history_p1 and history_p2:
        plot_training_history(history_p1, history_p2)
    else:
        print("  (Pass history objects to also plot training curves)")

    print(f"\nDone! All plots saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    evaluate()
