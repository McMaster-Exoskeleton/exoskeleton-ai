"""
FastAPI inference server for the exoskeleton TCN model.

Runs on the Raspberry Pi. The C++ control system sends a POST request to
/predict with a full sensor window and receives motor torque predictions
(in Nm) in response.

Usage:
    uvicorn scripts.server:app --host 0.0.0.0 --port 8000
    uvicorn scripts.server:app --host 127.0.0.1 --port 8000  # localhost only

Select model via query param (default: full):
    POST /predict?model=full
    POST /predict?model=single_joint

Dependencies (not in pyproject.toml, install on Pi):
    pip install fastapi uvicorn[standard] onnxruntime numpy msgpack

Input shape:  (1, window_size, feature_count)  — per model registry entry
Output shape: (1, window_size, output_count)   — server returns only the final timestep's values,
              denormalized to physical torque units (Nm)
"""

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import msgpack
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

JOINT_NAMES = ["hip_left", "hip_right", "knee_left", "knee_right"]

NORM_STATS_PATH = "data/processed/phase1/normalization_stats.json"
PILOTS_DIR = Path("configs/pilots")

MODEL_REGISTRY: dict[str, dict] = {
    "full": {
        "onnx_path": "outputs/model.onnx",
        "window_size": 187,
        "feature_count": 28,
        "joint_indices": [0, 1, 2, 3],
    },
    "single_joint": {
        "onnx_path": "outputs/model_single_joint.onnx",
        "window_size": 187,
        "feature_count": 7,
        "joint_indices": [0],  # hip_left only
    },
}
DEFAULT_MODEL = "full"

# ---------------------------------------------------------------------------
# App lifespan: load models and normalization stats once at startup
# ---------------------------------------------------------------------------

_targets_mean: np.ndarray | None = None
_targets_std: np.ndarray | None = None
_pilot_mass_kg: float = 1.0
_sessions: dict[str, ort.InferenceSession] = {}
_input_names: dict[str, str] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _targets_mean, _targets_std, _pilot_mass_kg, _sessions, _input_names

    # Load pilot config
    pilot_name = os.environ.get("PILOT", "default")
    pilot_path = PILOTS_DIR / f"{pilot_name}.json"
    if not pilot_path.exists():
        print(f"WARNING: Pilot config '{pilot_path}' not found, falling back to default.json")
        pilot_path = PILOTS_DIR / "default.json"
    pilot_cfg = json.loads(pilot_path.read_text())
    _pilot_mass_kg = float(pilot_cfg["mass_kg"])
    print(f"Pilot: {pilot_cfg.get('name', pilot_name)} | mass = {_pilot_mass_kg} kg")

    # Load normalization stats
    stats_path = Path(NORM_STATS_PATH)
    if not stats_path.exists():
        raise RuntimeError(f"Normalization stats not found: {stats_path.resolve()}")
    stats = json.loads(stats_path.read_text())
    _targets_mean = np.array(stats["targets"]["mean"], dtype=np.float32)
    _targets_std = np.array(stats["targets"]["std"], dtype=np.float32)
    print(f"Loaded normalization stats from {stats_path.resolve()}")

    # Load all registered model sessions
    for name, cfg in MODEL_REGISTRY.items():
        model_path = Path(cfg["onnx_path"])
        if not model_path.exists():
            print(f"WARNING: Model '{name}' not found at {model_path.resolve()}, skipping.")
            continue
        print(f"Loading ONNX model '{name}': {model_path.resolve()}")
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        _sessions[name] = session
        _input_names[name] = session.get_inputs()[0].name
        print(
            f"  Loaded '{name}': input='{_input_names[name]}' "
            f"shape=(1, {cfg['window_size']}, {cfg['feature_count']})"
        )

    if DEFAULT_MODEL not in _sessions:
        raise RuntimeError(
            f"Default model '{DEFAULT_MODEL}' could not be loaded. "
            "Ensure the ONNX file exists before starting the server."
        )

    yield

    _sessions.clear()
    _input_names.clear()
    print("Server shutting down.")


app = FastAPI(
    title="Exoskeleton TCN Inference Server",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------


def _run_inference(model_name: str, window: np.ndarray) -> dict:
    """
    Run inference with the named model and return a dict of denormalized predictions.

    Args:
        model_name: Key into MODEL_REGISTRY / _sessions.
        window: Input array of shape (1, window_size, feature_count).

    Returns:
        Dict with joint name keys (Nm values) and 'inference_ms'.
    """
    cfg = MODEL_REGISTRY[model_name]
    session = _sessions[model_name]
    input_name = _input_names[model_name]

    t0 = time.perf_counter()
    outputs = session.run(None, {input_name: window})
    inference_ms = (time.perf_counter() - t0) * 1000

    preds_norm = outputs[0][0, -1, :]  # shape: (output_count,)

    # Denormalize from z-score → Nm/kg, then scale by pilot mass → Nm
    indices = cfg["joint_indices"]
    preds_nm_per_kg = preds_norm * _targets_std[indices] + _targets_mean[indices]  # type: ignore[index]
    preds_phys = preds_nm_per_kg * _pilot_mass_kg

    result: dict = {JOINT_NAMES[j]: float(preds_phys[i]) for i, j in enumerate(indices)}
    result["inference_ms"] = round(inference_ms, 3)
    return result


def _resolve_model(model: str) -> str:
    """Validate the model query param and return it, or raise HTTP 400/503."""
    if model not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model}'. Available: {list(MODEL_REGISTRY.keys())}",
        )
    if model not in _sessions:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model}' is registered but not loaded (ONNX file missing at startup).",
        )
    return model


# ---------------------------------------------------------------------------
# Request schema (validated dynamically against the selected model)
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    window: list[list[float]] = Field(description="Sensor window: timesteps × features")

    # model_name is injected before validation by the endpoint
    model_config = {"arbitrary_types_allowed": True}

    # Stored after endpoint sets it before calling model_validate
    _model_name: str = "full"

    @model_validator(mode="after")
    def validate_shape(self) -> "PredictRequest":
        # Use the model name stored on the instance (set by endpoint before validation)
        model_name = getattr(self, "_model_name", DEFAULT_MODEL)
        cfg = MODEL_REGISTRY.get(model_name, MODEL_REGISTRY[DEFAULT_MODEL])
        window_size = cfg["window_size"]
        feature_count = cfg["feature_count"]

        rows = len(self.window)
        if rows != window_size:
            raise ValueError(f"Expected {window_size} timesteps, got {rows}")
        for i, row in enumerate(self.window):
            if len(row) != feature_count:
                raise ValueError(
                    f"Timestep {i}: expected {feature_count} features, got {len(row)}"
                )
        return self


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    """Quick liveness check — the C++ side can poll this on startup."""
    if DEFAULT_MODEL not in _sessions:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "ok",
        "models_loaded": list(_sessions.keys()),
        "pilot_mass_kg": _pilot_mass_kg,
    }


@app.post("/predict")
def predict(req: PredictRequest, model: str = DEFAULT_MODEL) -> dict:
    """
    JSON inference endpoint.

    Query params:
        model: Model name from MODEL_REGISTRY (default: 'full')

    Request body:
        {"window": [[f0..fN], ...]}  — shape (window_size, feature_count)

    Response:
        {"hip_left": <Nm>, ..., "inference_ms": <ms>}
    """
    model_name = _resolve_model(model)

    cfg = MODEL_REGISTRY[model_name]
    # Validate shape against the selected model
    rows = len(req.window)
    if rows != cfg["window_size"]:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {cfg['window_size']} timesteps for model '{model_name}', got {rows}",
        )
    for i, row in enumerate(req.window):
        if len(row) != cfg["feature_count"]:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Timestep {i}: expected {cfg['feature_count']} features "
                    f"for model '{model_name}', got {len(row)}"
                ),
            )

    input_array = np.array(req.window, dtype=np.float32)[np.newaxis, ...]
    return _run_inference(model_name, input_array)


@app.post("/predict_msgpack")
async def predict_msgpack(request: Request, model: str = DEFAULT_MODEL) -> Response:
    """
    Msgpack endpoint for lower-overhead inference.

    Query params:
        model: Model name from MODEL_REGISTRY (default: 'full')

    Request body:
        msgpack-encoded flat list of (window_size * feature_count) float32 values (row-major)

    Response body:
        msgpack-encoded dict {"hip_left": <Nm>, ..., "inference_ms": <ms>}
    """
    model_name = _resolve_model(model)
    cfg = MODEL_REGISTRY[model_name]

    if _targets_mean is None:
        raise HTTPException(status_code=503, detail="Normalization stats not loaded")

    raw = await request.body()
    flat = msgpack.unpackb(raw, raw=False)
    expected_len = cfg["window_size"] * cfg["feature_count"]
    if len(flat) != expected_len:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Expected {expected_len} values for model '{model_name}' "
                f"({cfg['window_size']} × {cfg['feature_count']}), got {len(flat)}"
            ),
        )

    input_array = np.array(flat, dtype=np.float32).reshape(
        1, cfg["window_size"], cfg["feature_count"]
    )
    result = _run_inference(model_name, input_array)
    return Response(content=msgpack.packb(result), media_type="application/x-msgpack")
