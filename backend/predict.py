"""
backend/predict.py
--------------
Run inference on a single image or an entire folder.

Usage
-----
  python backend/predict.py --image path/to/mri.jpg
  python backend/predict.py --folder path/to/folder/
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from config import CLASSES, IMG_SIZE, CHECKPOINTS, MODEL_NAME, RESULTS_DIR
from backend.load_compat import load_model_compat


# ─── Load model (cached) ──────────────────────────────────────────────────────

_model = None

def get_model() -> tf.keras.Model:
    global _model
    if _model is None:
        path = os.path.join(CHECKPOINTS, f"{MODEL_NAME}_final.keras")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model not found at: {path}\n"
                "Train the model first with:  python backend/train.py"
            )
        _model = load_model_compat(path)
        print(f"Model loaded from: {path}\n")
    return _model


# ─── Inference ────────────────────────────────────────────────────────────────

def preprocess(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)   # Keep [0, 255] — EfficientNet preprocesses internally
    return np.expand_dims(arr, 0)           # (1, H, W, 3)


def predict_single(image_path: str, verbose: bool = True) -> dict:
    model  = get_model()
    tensor = preprocess(image_path)
    probs  = model.predict(tensor, verbose=0)[0]
    idx    = int(np.argmax(probs))
    result = {
        "file":          os.path.basename(image_path),
        "predicted":     CLASSES[idx],
        "confidence":    float(probs[idx]),
        "probabilities": {cls: float(p) for cls, p in zip(CLASSES, probs)},
    }
    if verbose:
        _print_result(result)
    return result


def _print_result(r: dict):
    bar = "#" * int(r["confidence"] * 30)
    print(f"\n[{r['file']}]")
    print(f"   Prediction : {r['predicted'].upper()}")
    print(f"   Confidence : {r['confidence']*100:.2f}%  |{bar}|")
    print("   Per-class probabilities:")
    for cls, p in r["probabilities"].items():
        marker = " <--" if cls == r["predicted"] else ""
        print(f"     {cls:<15} {p*100:6.2f}%{marker}")


# ─── Batch & visualization ────────────────────────────────────────────────────

COLORS = {"glioma": "#E74C3C", "meningioma": "#F39C12",
          "notumor": "#2ECC71", "pituitary": "#3498DB"}


def predict_folder(folder: str, save_grid: bool = True):
    exts   = {".jpg", ".jpeg", ".png"}
    paths  = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in exts
    ]
    if not paths:
        print("No images found in folder.")
        return []

    results = [predict_single(p, verbose=True) for p in paths[:20]]  # cap at 20

    if save_grid:
        _save_grid(paths[:min(9, len(paths))], results[:min(9, len(results))])

    return results


def _save_grid(paths, results):
    n   = len(paths)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.array(axes).flatten()

    for ax, path, res in zip(axes, paths, results):
        img = Image.open(path).convert("RGB").resize(IMG_SIZE)
        ax.imshow(img)
        color = COLORS.get(res["predicted"], "white")
        ax.set_title(
            f"{res['predicted'].upper()}\n{res['confidence']*100:.1f}%",
            color="white", fontsize=10, fontweight="bold",
            bbox=dict(facecolor=color, alpha=0.8, pad=3, edgecolor="none"),
        )
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    patches = [mpatches.Patch(color=c, label=cls) for cls, c in COLORS.items()]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=9)
    fig.suptitle("Brain Tumor MRI - Predictions", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    out = os.path.join(RESULTS_DIR, "predictions_grid.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPrediction grid saved -> {out}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brain Tumor MRI Predictor")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  type=str, help="Path to a single MRI image")
    group.add_argument("--folder", type=str, help="Path to a folder of MRI images")
    args = parser.parse_args()

    if args.image:
        predict_single(args.image)
    else:
        predict_folder(args.folder)
