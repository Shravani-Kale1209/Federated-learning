"""
backend/admin_server.py
----------------------
Central aggregator and admin supervisor for federated learning.
Supports sample-weighted FedAvg, Multi-Krum + trimmed-mean deltas, optional RECESS-style probes.
"""

import os
import sys
import uuid
import pickle
import secrets

import numpy as np
from fastapi import FastAPI, Request, UploadFile, File, Response, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    MODEL_NAME,
    CHECKPOINTS,
    CLASSES,
    INPUT_SHAPE,
    FL_AGGREGATOR,
    FL_MIN_CLIENTS_FOR_KRUM,
    FL_KRUM_MULTI_K,
    FL_KRUM_NEIGHBOR_M,
    FL_TRIM_BETA,
    FL_MAX_CLIENTS,
    FL_RECESS_ENABLED,
    FL_RECESS_REQUIRED,
    FL_RECESS_TOLERANCE,
)
from backend.load_compat import load_model_compat
from backend.predict import preprocess
from backend.aggregation import aggregate_updates

app = FastAPI(title="Global Admin Server")


class AdminServerState:
    def __init__(self):
        self.model = None
        self.global_weights = None
        self.client_updates = []
        self.round_num = 1
        self.MAX_CLIENTS = FL_MAX_CLIENTS
        self.last_agg_num_samples = None
        self.last_audit: dict = {}
        self.pending_recess: dict[str, np.ndarray] = {}
        self.valid_recess_tokens: set[str] = set()


state = AdminServerState()


@app.on_event("startup")
def startup_event():
    print("═" * 60)
    print("  Global Admin Supervisor Starting (Port 8000)")
    print(f"  Aggregator: {FL_AGGREGATOR} | max_clients={state.MAX_CLIENTS}")
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


@app.get("/")
def admin_dashboard():
    dash_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "frontend", "templates", "admin_dashboard.html")
    )
    if os.path.exists(dash_path):
        with open(dash_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("Admin Dashboard building...", status_code=404)


@app.get("/api/admin_stats")
def get_stats():
    out = {
        "round_num": state.round_num,
        "clients_received": len(state.client_updates),
        "max_clients": state.MAX_CLIENTS,
        "aggregation": FL_AGGREGATOR,
    }
    if state.last_audit:
        out["last_audit"] = state.last_audit.copy()
        if isinstance(state.last_audit.get("clients_dropped"), list):
            out["clients_blocked_krum"] = len(state.last_audit["clients_dropped"])
    if state.last_agg_num_samples:
        total = sum(state.last_agg_num_samples)
        if total > 0:
            out["contributions"] = [
                {"name": f"Hospital {i + 1}", "value": round(100.0 * ns / total, 2)}
                for i, ns in enumerate(state.last_agg_num_samples)
            ]
        out["last_num_samples_per_client"] = list(state.last_agg_num_samples)
    out["recess"] = {
        "enabled": FL_RECESS_ENABLED,
        "required": FL_RECESS_REQUIRED,
        "pending_challenges": len(state.pending_recess),
        "validated_tokens_ready": len(state.valid_recess_tokens),
    }
    return out


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
            "probabilities": {cls: float(p) for cls, p in zip(CLASSES, probs)},
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return result


@app.get("/weights")
def get_weights():
    """Hospitals call this to GET the model."""
    if state.global_weights is None:
        return Response(content="Model not initialized", status_code=503)
    weights_bytes = pickle.dumps(state.global_weights)
    print(f"[ROUND {state.round_num}] Distributed global weights to a Hospital.")
    return Response(content=weights_bytes, media_type="application/octet-stream")


@app.get("/verification/challenge")
def verification_challenge():
    """RECESS-style: issue probe input + token; hospitals respond with probs from their model."""
    if not FL_RECESS_ENABLED or state.model is None:
        raise HTTPException(status_code=404, detail="RECESS verification disabled")
    seed = int.from_bytes(secrets.token_bytes(8), "little", signed=False) % (2**31)
    rng = np.random.default_rng(seed)
    inp = rng.standard_normal(size=(1, *INPUT_SHAPE), dtype=np.float32)
    inp = np.clip(inp, -2.5, 2.5).astype(np.float32)
    ref = state.model.predict(inp, verbose=0)[0].astype(np.float64)

    token = str(uuid.uuid4())
    state.pending_recess[token] = ref
    payload = pickle.dumps({"token": token, "input": inp})
    return Response(content=payload, media_type="application/octet-stream")


@app.post("/verification/respond")
async def verification_respond(request: Request):
    if not FL_RECESS_ENABLED:
        raise HTTPException(status_code=404, detail="RECESS verification disabled")

    obj = pickle.loads(await request.body())
    token = obj.get("token")
    probs = np.asarray(obj.get("probs"), dtype=np.float64)
    ref = state.pending_recess.pop(token, None)

    if ref is None:
        raise HTTPException(status_code=400, detail="Unknown or expired challenge token")

    if probs.shape != ref.shape:
        raise HTTPException(status_code=400, detail="Probability shape mismatch")

    dist_l2 = float(np.linalg.norm(probs - ref))
    audit = {"accepted": False, "l2_vs_global": round(dist_l2, 6), "threshold": FL_RECESS_TOLERANCE}

    if dist_l2 <= FL_RECESS_TOLERANCE:
        state.valid_recess_tokens.add(token)
        audit["accepted"] = True
        print(f"[RECESS] Token {token[:8]}... accepted (l2={dist_l2:.4f}).")
        return {"status": "ok", "verification": audit}

    print(f"[RECESS] Token {token[:8]}... rejected (l2={dist_l2:.4f}).")
    return {"status": "rejected", "verification": audit}


def _parse_weight_payload(body: bytes):
    """Optional dict payload with weights + num_samples + verification_token."""
    obj = pickle.loads(body)
    if isinstance(obj, dict) and "weights" in obj:
        w = obj["weights"]
        n = int(obj.get("num_samples", 1))
        token = obj.get("verification_token")
    else:
        w = obj
        n = 1
        token = None
    if n < 1:
        n = 1
    return w, n, token


@app.post("/weights")
async def receive_weights(request: Request):
    """Hospitals POST learned weights; optionally RECESS token when required."""
    body = await request.body()
    local_weights, num_samples, vtoken = _parse_weight_payload(body)

    if FL_RECESS_REQUIRED:
        if not vtoken or vtoken not in state.valid_recess_tokens:
            raise HTTPException(
                status_code=403,
                detail="RECESS verification required: call /verification/challenge then /verification/respond",
            )
        state.valid_recess_tokens.discard(vtoken)

    state.client_updates.append((local_weights, num_samples))

    num_received = len(state.client_updates)
    print(
        f"[ROUND {state.round_num}] Secure payload received from hospital "
        f"(num_samples={num_samples}). ({num_received}/{state.MAX_CLIENTS})"
    )

    if num_received >= state.MAX_CLIENTS:
        aggregate_and_update()

    return {"status": "ok", "message": "Weights securely received."}


def aggregate_and_update():
    clients = state.client_updates
    num_samples_list = [n for _, n in clients]
    state.last_agg_num_samples = tuple(num_samples_list)
    state.valid_recess_tokens.clear()
    state.pending_recess.clear()

    mode = FL_AGGREGATOR.strip().lower()
    if mode not in ("weighted_fedavg", "krum_trimmed_mean", "trimmed_mean"):
        print(f"Unknown FL_AGGREGATOR={mode!r}, falling back to weighted_fedavg")
        mode = "weighted_fedavg"

    print(
        f"\n[ROUND {state.round_num}] Aggregating {state.MAX_CLIENTS} hospital update(s) "
        f"using mode={mode!r}..."
    )

    new_w, audit = aggregate_updates(
        state.global_weights,
        clients,
        mode,  # type: ignore[arg-type]
        min_clients_for_krum=FL_MIN_CLIENTS_FOR_KRUM,
        krum_multi_k=FL_KRUM_MULTI_K,
        krum_neighbor_m=FL_KRUM_NEIGHBOR_M,
        trim_beta_per_tail=FL_TRIM_BETA,
    )
    total_n = float(sum(num_samples_list))
    audit["sample_weights_fraction"] = [round(n / total_n, 4) for n in num_samples_list]
    audit["recess_was_required_last_round"] = FL_RECESS_REQUIRED
    state.last_audit = audit

    state.global_weights = new_w
    if state.model:
        state.model.set_weights(new_w)
    state.client_updates = []
    state.round_num += 1

    pct = ", ".join(f"{100.0 * n / total_n:.1f}%" for n in num_samples_list)
    print(f"[ROUND {state.round_num - 1}] Global model updated (sample fractions: {pct}).")
    dropped = audit.get("clients_dropped") or []
    if dropped:
        print(f"  Robust filter dropped client indices (0-based): {dropped}")
    print("═" * 60)


if __name__ == "__main__":
    uvicorn.run("backend.admin_server:app", host="127.0.0.1", port=8000, reload=True)
