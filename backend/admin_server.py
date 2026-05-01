"""
backend/admin_server.py
----------------------
The Central Aggregator and Admin Supervisor for Federated Learning.
Runs on Port 8000.
Handles only FedAvg logic and Admin monitoring.
"""

import os
import sys
import pickle
import numpy as np
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, Response
import uvicorn
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import MODEL_NAME, CHECKPOINTS, CLASSES
from backend.load_compat import load_model_compat
from backend.predict import preprocess

app = FastAPI(title="Global Admin Server")

class AdminServerState:
    def __init__(self):
        self.model = None
        self.global_weights = None
        self.client_weights_list = []
        self.round_num = 1
        self.MAX_CLIENTS = 2

state = AdminServerState()

@app.on_event("startup")
def startup_event():
    print("═" * 60)
    print("  Global Admin Supervisor Starting (Port 8000)")
    print("═" * 60)
    
    model_path = os.path.join(CHECKPOINTS, f"{MODEL_NAME}_final.keras")
    if os.path.exists(model_path):
        print(f"Loading base global model from: {model_path}")
        model = load_model_compat(model_path)
        state.model = model
        state.global_weights = model.get_weights()
    else:
        print("ERROR: Could not find base model in checkboxes/")
        sys.exit(1)

# ─── Admin Dashboard ──────────────────────────────────────────────────────────

@app.get("/")
def admin_dashboard():
    dash_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "templates", "admin_dashboard.html"))
    if os.path.exists(dash_path):
        with open(dash_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("Admin Dashboard building...", status_code=404)

@app.get("/api/admin_stats")
def get_stats():
    return {
        "round_num": state.round_num,
        "clients_received": len(state.client_weights_list),
        "max_clients": state.MAX_CLIENTS
    }

@app.post("/api/predict")
async def admin_predict(file: UploadFile = File(...)):
    """Admin tests the live Global Model natively."""
    if not state.model:
        return {"error": "Server baseline model not initialized"}
        
    temp_path = os.path.join(os.path.dirname(__file__), "admin_temp_upload.jpg")
    content = await file.read()
    with open(temp_path, "wb") as f:
        f.write(content)
        
    try:
        tensor = preprocess(temp_path)
        probs = state.model.predict(tensor, verbose=0)[0]
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

# ─── Decentralized Hooks ──────────────────────────────────────────────────────

@app.get("/weights")
def get_weights():
    """Hospitals call this to GET the model."""
    if state.global_weights is None:
        return Response(content="Model not initialized", status_code=503)
    weights_bytes = pickle.dumps(state.global_weights)
    print(f"[ROUND {state.round_num}] Distributed global weights to a Hospital.")
    return Response(content=weights_bytes, media_type="application/octet-stream")

@app.post("/weights")
async def receive_weights(request: Request):
    """Hospitals securely POST their learned weight matrices here."""
    body = await request.body()
    local_weights = pickle.loads(body)
    state.client_weights_list.append(local_weights)
    
    num_received = len(state.client_weights_list)
    print(f"[ROUND {state.round_num}] Secure payload received from hospital. ({num_received}/{state.MAX_CLIENTS})")
    
    if num_received >= state.MAX_CLIENTS:
        aggregate_and_update()
        
    return {"status": "ok", "message": "Weights securely received."}

def aggregate_and_update():
    print(f"\n[ROUND {state.round_num}] Aggregating weights from {state.MAX_CLIENTS} hospitals...")
    new_global_weights = []
    
    for layer_idx in range(len(state.client_weights_list[0])):
        layer_matrices = [client_weights[layer_idx] for client_weights in state.client_weights_list]
        layer_mean = np.mean(layer_matrices, axis=0)
        new_global_weights.append(layer_mean)
        
    state.global_weights = new_global_weights
    if state.model:
        state.model.set_weights(new_global_weights) 
    state.client_weights_list = []
    state.round_num += 1
    
    print(f"[ROUND {state.round_num-1}] Global Model updated natively via FedAvg!")
    print("═" * 60)

if __name__ == "__main__":
    uvicorn.run("backend.admin_server:app", host="127.0.0.1", port=8000, reload=True)
