"""
setup_check.py
--------------
Run this BEFORE training to verify:
  ✓ TensorFlow is installed correctly
  ✓ GPU is detected
  ✓ Dataset is in the right folder structure
  ✓ All classes are present with enough images
  ✓ A sample image loads and preprocesses correctly
"""

import os
import sys

print("=" * 60)
print("  Brain Tumor MRI Classifier — Setup Check")
print("=" * 60)

errors   = []
warnings = []

# ── 1. TensorFlow ─────────────────────────────────────────────────────────────
print("\n[1/5] Checking TensorFlow …")
try:
    import tensorflow as tf
    print(f"  ✓ TensorFlow {tf.__version__}")
except ImportError:
    errors.append("TensorFlow not installed. Run:  pip install -r requirements.txt")

# ── 2. GPU ────────────────────────────────────────────────────────────────────
print("\n[2/5] Checking GPU …")
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for g in gpus:
        print(f"  ✓ GPU found: {g.name}")
else:
    warnings.append("No GPU detected. Training will be slow (~4-8 hrs on CPU). "
                    "Use Google Colab for free GPU.")
    print("  ⚠  No GPU detected")

# ── 3. Dataset structure ──────────────────────────────────────────────────────
print("\n[3/5] Checking dataset structure …")
from config import TRAIN_DIR, TEST_DIR, CLASSES

EXPECTED_MIN_IMAGES = 50   # per class

for split_name, split_dir in [("Training", TRAIN_DIR), ("Testing", TEST_DIR)]:
    if not os.path.isdir(split_dir):
        errors.append(
            f"Missing folder: {split_dir}\n"
            f"  → Download from Kaggle and place it at: {split_dir}"
        )
        continue

    print(f"\n  {split_name}/")
    for cls in CLASSES:
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir):
            errors.append(f"Missing class folder: {cls_dir}")
            continue
        images = [
            f for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        n = len(images)
        ok = "✓" if n >= EXPECTED_MIN_IMAGES else "⚠"
        print(f"    {ok}  {cls:<15} {n:>5} images")
        if n < EXPECTED_MIN_IMAGES:
            warnings.append(
                f"{split_name}/{cls} has only {n} images (expected ≥ {EXPECTED_MIN_IMAGES})"
            )

# ── 4. Sample image load ──────────────────────────────────────────────────────
print("\n[4/5] Testing image loading …")
try:
    import numpy as np
    from PIL import Image

    # Find any image in the dataset
    sample = None
    for cls in CLASSES:
        cls_dir = os.path.join(TRAIN_DIR, cls)
        if os.path.isdir(cls_dir):
            for f in os.listdir(cls_dir):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    sample = os.path.join(cls_dir, f)
                    break
        if sample:
            break

    if sample:
        img = Image.open(sample).convert("RGB").resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        print(f"  ✓ Sample image loaded: {os.path.basename(sample)}")
        print(f"    Shape: {arr.shape}  Min: {arr.min():.3f}  Max: {arr.max():.3f}")
    else:
        warnings.append("Could not find any image in the dataset to test loading.")
except Exception as e:
    errors.append(f"Image loading failed: {e}")

# ── 5. Other imports ──────────────────────────────────────────────────────────
print("\n[5/5] Checking other dependencies …")
deps = [
    ("numpy",       "numpy"),
    ("matplotlib",  "matplotlib"),
    ("seaborn",     "seaborn"),
    ("sklearn",     "scikit-learn"),
    ("PIL",         "Pillow"),
    ("tqdm",        "tqdm"),
    ("cv2",         "opencv-python"),
    ("pandas",      "pandas"),
]
for module, pkg in deps:
    try:
        __import__(module)
        print(f"  ✓ {pkg}")
    except ImportError:
        errors.append(f"Missing package: {pkg} — run:  pip install {pkg}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
if not errors and not warnings:
    print("  ✅  All checks passed! You're ready to train.")
    print("\n  Run:  python backend/train.py")
else:
    if warnings:
        print(f"\n  ⚠  {len(warnings)} warning(s):")
        for w in warnings:
            print(f"     • {w}")
    if errors:
        print(f"\n  ❌  {len(errors)} error(s) — fix these before training:")
        for e in errors:
            print(f"     • {e}")
        sys.exit(1)

print("=" * 60)
