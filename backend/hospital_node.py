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
import pickle
import requests
import numpy as np
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    CHECKPOINTS,
    MODEL_NAME,
    IMG_SIZE,
    BATCH_SIZE,
    CLASSES,
    FL_FEDPROX_MU,
    HOSPITAL_RECESS,
)
from backend.load_compat import load_model_compat
from backend.predict import preprocess

app = FastAPI(title="Local Hospital Node")

# --- CORS MIDDLEWARE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

hospital_state = {
    "name": "Hospital X", 
    "org_id": "UNKNOWN",
    "password": "node_password",
    "status": "Idle", 
    "model": None,
    "token": None
}
ADMIN_URL = "http://127.0.0.1:8000"
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")


def _count_training_images(root_dir: str) -> int:
    n = 0
    for _, _, files in os.walk(root_dir):
        for fname in files:
            if fname.lower().endswith(_IMAGE_EXTS):
                n += 1
    return max(n, 1)

def get_auth_token():
    """Authenticates with the Admin server to get a JWT token."""
    try:
        res = requests.post(f"{ADMIN_URL}/api/auth/login", json={
            "username": hospital_state["org_id"],
            "password": hospital_state["password"]
        })
        if res.status_code == 200:
            hospital_state["token"] = res.json().get("token")
            print(f"Authenticated successfully as {hospital_state['org_id']}")
            return True
        else:
            print(f"Authentication failed: {res.text}")
    except Exception as e:
        print(f"Error connecting to admin for auth: {e}")
    return False

@app.on_event("startup")
def startup_event():
    model_path = os.path.join(CHECKPOINTS, f"{MODEL_NAME}_final.keras")
    if os.path.exists(model_path):
        model = load_model_compat(model_path)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        hospital_state["model"] = model
        
        # Initial sync
        if get_auth_token():
            try:
                headers = {"Authorization": f"Bearer {hospital_state['token']}"}
                res = requests.get(f"{ADMIN_URL}/weights", headers=headers)
                if res.status_code == 200:
                    global_weights = pickle.loads(res.content)
                    model.set_weights(global_weights)
                    print("Synchronized global weights on boot.")
            except Exception as e:
                print(f"Sync failed: {e}")
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
            if not hospital_state["token"]:
                get_auth_token()
            
            headers = {"Authorization": f"Bearer {hospital_state['token']}"}
            res = requests.get(f"{ADMIN_URL}/weights", headers=headers)
            res.raise_for_status()
            global_weights = pickle.loads(res.content)
            model.set_weights(global_weights)
        except Exception as e:
            print(f"Failed to fetch global weights: {e}")
            # We can still proceed to train locally if offline
            
        hospital_state["status"] = "Training locally on patient data..."
        
        # 3. Create a Local tf.data Pipeline from the extracted folder
        local_ds = tf.keras.utils.image_dataset_from_directory(
            temp_dir,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode='int'
        )
        num_samples = _count_training_images(temp_dir)

        # 3b. Train locally (FedProx tether to global snapshot when mu > 0)
        anchor = [tf.constant(w, dtype=tf.float32) for w in model.get_weights()]
        if FL_FEDPROX_MU and FL_FEDPROX_MU > 0:
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
            opt = tf.keras.optimizers.Adam(1e-5)
            mu = float(FL_FEDPROX_MU)
            for _ in range(1):
                for batch_x, batch_y in local_ds:
                    with tf.GradientTape() as tape:
                        preds = model(batch_x, training=True)
                        cls_loss = loss_fn(batch_y, preds)
                        prox = tf.cast(0.0, dtype=tf.float32)
                        for i, vw in enumerate(model.weights):
                            prox = prox + tf.reduce_sum(tf.square(vw - anchor[i]))
                        total_loss = cls_loss + (mu / 2.0) * prox
                    grads = tape.gradient(total_loss, model.trainable_variables)
                    opt.apply_gradients(zip(grads, model.trainable_variables))
        else:
            model.fit(local_ds, epochs=1, verbose=1)

        hospital_state["status"] = "Securely streaming parameters to Admin..."

        # 4. Optional RECESS verify (must succeed before POST if admin requires token)
        verification_token = None
        if HOSPITAL_RECESS:
            try:
                cr = requests.get(f"{ADMIN_URL}/verification/challenge", timeout=30)
                if cr.status_code == 200:
                    pack = pickle.loads(cr.content)
                    token = pack["token"]
                    inp = pack["input"]
                    pr = hospital_state["model"].predict(inp, verbose=0)[0]
                    rr = requests.post(
                        f"{ADMIN_URL}/verification/respond",
                        data=pickle.dumps(
                            {"token": token, "probs": pr.astype(np.float32)}
                        ),
                        timeout=30,
                    )
                    if rr.ok and rr.json().get("status") == "ok":
                        verification_token = token
                    else:
                        print(f"[RECESS] Verification not accepted: {rr.text}")
            except Exception as e:
                print(f"[RECESS] Challenge/response skipped: {e}")

        # 5. Send weights (+ optional verification token)
        new_weights = model.get_weights()
        payload_dict = {"weights": new_weights, "num_samples": num_samples}
        if verification_token:
            payload_dict["verification_token"] = verification_token
        payload = pickle.dumps(payload_dict)
        
        try:
            if not hospital_state["token"]:
                get_auth_token()
            
            headers = {"Authorization": f"Bearer {hospital_state['token']}"}
            res = requests.post(f"{ADMIN_URL}/weights", data=payload, headers=headers)
            
            if res.status_code == 401:
                # Token might be stale, retry once
                print("Token invalid, retrying authentication...")
                if get_auth_token():
                    headers = {"Authorization": f"Bearer {hospital_state['token']}"}
                    res = requests.post(f"{ADMIN_URL}/weights", data=payload, headers=headers)
            
            res.raise_for_status()
            hospital_state["status"] = "Federated Cycle Complete! Patient data erased."
        except Exception as e:
            print(f"Upload failed: {e}")
            error_detail = getattr(res, 'text', str(e)) if 'res' in locals() else str(e)
            hospital_state["status"] = f"Upload failed: {error_detail}"
            
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
    parser.add_argument("--org_id", type=str, required=True)
    parser.add_argument("--password", type=str, default="node_password")
    args = parser.parse_args()
    
    hospital_state["name"] = args.name
    hospital_state["org_id"] = args.org_id
    hospital_state["password"] = args.password
    
    uvicorn.run(app, host="127.0.0.1", port=args.port)
