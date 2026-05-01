# Brain Tumor MRI Classifier 🧠

A deep learning model that classifies brain MRI scans into **4 categories**:
`Glioma` · `Meningioma` · `No Tumor` · `Pituitary`

Built with **EfficientNetB0 Transfer Learning** — trains to ~95%+ accuracy in 30–60 minutes on a free GPU.

---

## Project Structure

```
codeShield_training/
├── config.py              ← All hyperparameters and paths
├── requirements.txt       ← Python dependencies
│
├── backend/
│   ├── data_loader.py     ← tf.data pipeline + augmentation
│   ├── model.py           ← EfficientNetB0 model builder
│   ├── train.py           ← Two-phase training script
│   ├── evaluate.py        ← Metrics + plots
│   └── predict.py         ← Inference CLI
│
├── notebooks/
│   └── colab_train.py     ← Copy-paste into Google Colab
│
├── dataset/               ← PUT KAGGLE DATA HERE
│   ├── Training/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   └── Testing/
│       └── ... (same structure)
│
├── checkpoints/           ← Auto-created — saved models
├── results/               ← Auto-created — plots & metrics
└── logs/                  ← Auto-created — TensorBoard logs
```

---

## Quick Start

### Option A — Google Colab (Recommended — Free GPU!)

1. Open [Google Colab](https://colab.research.google.com/) → New notebook
2. Set runtime: **Runtime → Change runtime type → T4 GPU**
3. Copy the contents of `notebooks/colab_train.py` into cells
4. Upload the Kaggle dataset to your Google Drive
5. Run all cells — done in ~45 minutes

### Option B — Run Locally

#### Step 1: Download the Dataset

```bash
pip install kaggle
# Place your kaggle.json API key in ~/.kaggle/
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
unzip brain-tumor-mri-dataset.zip -d dataset/
```

Or download manually from:
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Unzip so that you have:
```
dataset/Training/glioma/
dataset/Training/meningioma/
dataset/Training/notumor/
dataset/Training/pituitary/
dataset/Testing/...
```

#### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

#### Step 3: Train

```bash
python backend/train.py
```

This runs two phases automatically:
- **Phase 1** (10 epochs): Trains only the classification head, base frozen
- **Phase 2** (10 epochs): Fine-tunes the top 30 layers of EfficientNetB0

Best model saved to `checkpoints/` after each phase.

#### Step 4: Evaluate

```bash
python backend/evaluate.py
```

Generates in `results/`:
- `confusion_matrix.png`
- `per_class_accuracy.png`
- `training_history.png`

#### Step 5: Predict

```bash
# Single image
python backend/predict.py --image path/to/mri_scan.jpg

# Entire folder
python backend/predict.py --folder path/to/folder/
```

---

## Model Architecture

```
Input (224×224×3)
    ↓
[Data Augmentation]  — flip, rotate, zoom, translate, brightness
    ↓
EfficientNetB0       — pretrained on ImageNet (1,280 feature maps)
    ↓
GlobalAveragePooling2D
    ↓
BatchNormalization
    ↓
Dense(256, relu) + L2 regularization
    ↓
Dropout(0.4)
    ↓
Dense(128, relu) + L2 regularization
    ↓
Dropout(0.3)
    ↓
Dense(4, softmax)    — glioma / meningioma / notumor / pituitary
```

### Two-Phase Training

| Phase | Base | LR | Epochs |
|-------|------|----|--------|
| 1 – Head only | Frozen | `1e-3` | 10 |
| 2 – Fine-tune | Top 30 layers unfrozen | `1e-5` | 10 |

---

## Expected Results

| Metric | Expected |
|--------|----------|
| Test Accuracy | **94–97%** |
| Training time (T4 GPU) | ~30–45 min |
| Training time (CPU only) | 4–8 hours |

---

## TensorBoard

```bash
tensorboard --logdir logs/
```

---

## Configuration

All settings are in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `IMG_SIZE` | `(224, 224)` | Input image size |
| `BATCH_SIZE` | `32` | Batch size |
| `EPOCHS_FREEZE` | `10` | Phase 1 epochs |
| `EPOCHS_UNFREEZE` | `10` | Phase 2 epochs |
| `LEARNING_RATE` | `1e-3` | Phase 1 LR |
| `FINE_TUNE_LR` | `1e-5` | Phase 2 LR |
| `CLASSES` | 4 classes | Tumor categories |
