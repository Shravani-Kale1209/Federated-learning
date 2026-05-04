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

# ─── Federated learning (aggregator + defenses) ────────────────────────────
# Aggregator modes: weighted_fedavg | krum_trimmed_mean | trimmed_mean
FL_AGGREGATOR = os.environ.get("FL_AGGREGATOR", "krum_trimmed_mean")
FL_MIN_CLIENTS_FOR_KRUM = int(os.environ.get("FL_MIN_CLIENTS_FOR_KRUM", "4"))
FL_KRUM_MULTI_K = max(1, int(os.environ.get("FL_KRUM_MULTI_K", "2")))
_krm = os.environ.get("FL_KRUM_NEIGHBOR_M")
FL_KRUM_NEIGHBOR_M = int(_krm) if _krm else None
FL_TRIM_BETA = float(os.environ.get("FL_TRIM_BETA", "0.1"))

FL_MAX_CLIENTS = max(2, int(os.environ.get("FL_MAX_CLIENTS", "2")))
# Set > 0 (e.g. 0.01) to enable FedProx custom training loop on hospital nodes.
FL_FEDPROX_MU = float(os.environ.get("FL_FEDPROX_MU", "0"))

FL_RECESS_ENABLED = os.environ.get("FL_RECESS_ENABLED", "1").lower() in ("1", "true", "yes")
FL_RECESS_REQUIRED = os.environ.get("FL_RECESS_REQUIRED", "0").lower() in ("1", "true", "yes")
# Relaxed default: Gaussian probes are OOD vs MRI-trained weights; tighten for stronger guarantees.
FL_RECESS_TOLERANCE = float(os.environ.get("FL_RECESS_TOLERANCE", "8.0"))

# Hospital node: run RECESS challenge/response before uploading (should match server requirement)
HOSPITAL_RECESS = os.environ.get("HOSPITAL_RECESS", "1").lower() in ("1", "true", "yes")
