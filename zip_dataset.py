"""
zip_dataset.py
--------------
Zips the dataset/ folder into dataset.zip for Colab upload.
Run from D:\codeShield_training\
"""

import zipfile
import os
from pathlib import Path

DATASET_DIR = Path("dataset")
OUTPUT_ZIP  = Path("dataset.zip")

if not DATASET_DIR.exists():
    print("ERROR: dataset/ folder not found. Run from D:\\codeShield_training\\")
    exit(1)

if OUTPUT_ZIP.exists():
    OUTPUT_ZIP.unlink()
    print("Removed old dataset.zip")

# Collect all files
all_files = list(DATASET_DIR.rglob("*"))
image_files = [f for f in all_files if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]

print(f"Found {len(image_files)} images to zip …")
print("Zipping ", end="", flush=True)

with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    for i, fpath in enumerate(image_files):
        zf.write(fpath, fpath)           # preserve dataset/... path inside zip
        if i % 200 == 0:
            print(".", end="", flush=True)

size_mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
print(f"\n\n✓ Done!  dataset.zip created ({size_mb:.1f} MB)")
print(f"  Location: {OUTPUT_ZIP.resolve()}")
print("\nNext: Upload dataset.zip to Google Colab.")
