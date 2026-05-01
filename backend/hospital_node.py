"""
backend/hospital_node.py
------------------------
Localized isolated hospital web node.
Allows local doctors to securely upload a .zip of MRI images.
Extracts locally, trains local model on that data, sends exactly the weights to Admin, and deletes patient data.
"""

import os
import sys
import shutil
import zipfile
import tempfile
import argparse
import asyncio
import pickle
import requests
import numpy as np
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse
import uvicorn
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import CHECKPOINTS, MODEL_NAME, IMG_SIZE, BATCH_SIZE, CLASSES
from backend.load_compat import load_model_compat
from backend.predict import preprocess

app = FastAPI(title="Local Hospital Node")
hospital_state = {"name": "Hospital X", "status": "Idle", "model": None}
ADMIN_URL = "http://127.0.0.1:8000"

@app.on_event("startup")
def startup_event():
    model_path = os.path.join(CHECKPOINTS, f"{MODEL_NAME}_final.keras")
    if os.path.exists(model_path):
        model = load_model_compat(model_path)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        hospital_state["model"] = model
        
        try:
            res = requests.get(f"{ADMIN_URL}/weights")
            if res.status_code == 200:
                global_weights = pickle.loads(res.content)
                model.set_weights(global_weights)
                print("Synchronized global weights on boot.")
        except Exception:
            print("Admin server invisible on boot. Using local checkpoint baseline.")
    else:
        print("CRITICAL: Base model not found in checkpoints/")

@app.get("/")
def hospital_dashboard():
    # Pass hospital name into the UI dynamically
    dash_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "templates", "hospital_dashboard.html"))
    if os.path.exists(dash_path):
        with open(dash_path, "r", encoding="utf-8") as f:
            html = f.read()
            return HTMLResponse(content=html.replace("{{HOSPITAL_NAME}}", hospital_state["name"]))
    return HTMLResponse("Dashboard building...", status_code=404)

@app.get("/api/status")
def get_status():
    return {"status": hospital_state["status"]}

def run_local_training(zip_path: str):
    hospital_state["status"] = "Extracting data..."
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 1. Unzip securely
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            
        hospital_state["status"] = "Fetching Global Intelligence..."
        
        # 2. Sync Latest Global Weights
        model = hospital_state["model"]
        if not model:
            hospital_state["status"] = "Error: Base model absent"
            return
            
        try:
            res = requests.get(f"{ADMIN_URL}/weights")
            res.raise_for_status()
            global_weights = pickle.loads(res.content)
            model.set_weights(global_weights)
        except Exception as e:
            print(f"Failed to fetch global weights: {e}")
            # We can still proceed to train locally if offline
            
        hospital_state["status"] = "Training locally on patient data..."
        
        # 3. Create a Local tf.data Pipeline from the extracted folder
        # Expecting zip to have folders for classes inside, or just images
        local_ds = tf.keras.utils.image_dataset_from_directory(
            temp_dir,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode='int' # Simplistic matching for demo
        )
        
        # Train!
        model.fit(local_ds, epochs=1, verbose=1)
        
        hospital_state["status"] = "Securely streaming parameters to Admin..."
        
        # 4. Extract new weights and send
        new_weights = model.get_weights()
        payload = pickle.dumps(new_weights)
        
        try:
            requests.post(f"{ADMIN_URL}/weights", data=payload)
            hospital_state["status"] = "Federated Cycle Complete! Patient data erased."
        except Exception as e:
            hospital_state["status"] = "Upload failed."
            
    finally:
        # 5. Clean up patient datasets (HIPPA compliance simulation)
        shutil.rmtree(temp_dir, ignore_errors=True)
        if os.path.exists(zip_path):
            os.remove(zip_path)


@app.post("/api/upload")
async def upload_patient_data(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.endswith(".zip"):
        return {"error": "Upload must be a .zip file containing MRI images."}
        
    temp_zip = os.path.join(tempfile.gettempdir(), file.filename)
    with open(temp_zip, "wb") as f:
        f.write(await file.read())
        
    hospital_state["status"] = "Data received. Initializing FL sequence..."
    background_tasks.add_task(run_local_training, temp_zip)
    
    return {"message": "Data securely ingested. Local training starting in background."}

@app.post("/api/predict")
async def local_predict(file: UploadFile = File(...)):
    """Local Hospital interface tests their local active model instance."""
    if not hospital_state["model"]:
        return {"error": "Local model uninitialized"}
        
    temp_path = os.path.join(tempfile.gettempdir(), "hosp_temp_upload.jpg")
    content = await file.read()
    with open(temp_path, "wb") as f:
        f.write(content)
        
    try:
        tensor = preprocess(temp_path)
        probs = hospital_state["model"].predict(tensor, verbose=0)[0]
        idx = int(np.argmax(probs))
        
        result = {
            "predicted": CLASSES[idx].upper(),
            "confidence": float(probs[idx]),
            "probabilities": {cls: float(p) for cls, p in zip(CLASSES, probs)}
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--name", type=str, required=True)
    args = parser.parse_args()
    
    hospital_state["name"] = args.name
    uvicorn.run(app, host="127.0.0.1", port=args.port)
