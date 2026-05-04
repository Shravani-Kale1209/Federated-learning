import os
import sys
import tensorflow as tf

sys.path.insert(0, os.getcwd())
print("Path set. Importing components...")
from backend.load_compat import load_model_compat
print("Imported load_model_compat")

MODEL_NAME = "brain_tumor_classifier"
CHECKPOINTS = "checkpoints"
model_path = os.path.join(CHECKPOINTS, f"{MODEL_NAME}_final.keras")

print(f"Loading model from {model_path}...")
model = load_model_compat(model_path)
print("Model loaded successfully!")
print(f"Weights shape: {[w.shape for w in model.get_weights()[:3]]}")
