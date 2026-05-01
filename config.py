"""
config.py
---------
Central configuration for the Brain Tumor MRI Classifier.
Edit DATASET_ROOT to point to your downloaded Kaggle dataset.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(BASE_DIR, "dataset")          # Kaggle data goes here
TRAIN_DIR    = os.path.join(DATASET_ROOT, "Training")
TEST_DIR     = os.path.join(DATASET_ROOT, "Testing")
CHECKPOINTS  = os.path.join(BASE_DIR, "checkpoints")
LOGS_DIR     = os.path.join(BASE_DIR, "logs")
RESULTS_DIR  = os.path.join(BASE_DIR, "results")

os.makedirs(CHECKPOINTS, exist_ok=True)
os.makedirs(LOGS_DIR,    exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Classes ──────────────────────────────────────────────────────────────────
CLASSES      = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES  = len(CLASSES)

# ─── Image settings ───────────────────────────────────────────────────────────
IMG_SIZE     = (224, 224)   # EfficientNetB0 native size
IMG_CHANNELS = 3
INPUT_SHAPE  = (*IMG_SIZE, IMG_CHANNELS)

# ─── Training hyperparameters ─────────────────────────────────────────────────
BATCH_SIZE       = 32
EPOCHS_FREEZE    = 10       # Phase 1 – only train head, base frozen
EPOCHS_UNFREEZE  = 10       # Phase 2 – fine-tune top layers
LEARNING_RATE    = 1e-3
FINE_TUNE_LR     = 1e-5

# ─── Augmentation ─────────────────────────────────────────────────────────────
ROTATION_RANGE      = 20
ZOOM_RANGE          = 0.2
WIDTH_SHIFT_RANGE   = 0.1
HEIGHT_SHIFT_RANGE  = 0.1
HORIZONTAL_FLIP     = True
BRIGHTNESS_RANGE    = (0.8, 1.2)

# ─── Misc ─────────────────────────────────────────────────────────────────────
SEED             = 42
VALIDATION_SPLIT = 0.2
MODEL_NAME       = "brain_tumor_classifier"
