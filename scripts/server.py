"""
FastAPI inference server for the exoskeleton TCN model.

Runs on the Raspberry Pi. The C++ control system sends a POST request to
/predict with a full (187, 28) window of sensor data and receives 4 motor
torque predictions in response.

Usage:
    uvicorn scripts.server:app --host 0.0.0.0 --port 8000
    uvicorn scripts.server:app --host 127.0.0.1 --port 8000  # localhost only

Dependencies (not in pyproject.toml, install on Pi):
    pip install fastapi uvicorn[standard] onnxruntime numpy msgpack

Input shape:  (1, WINDOW_SIZE, 28)  — WINDOW_SIZE defaults to 187 for standard TCN
Output shape: (1, WINDOW_SIZE, 4)   — server returns only the final timestep's 4 values
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import time

import msgpack
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = "outputs/model.onnx"

# Must match the effective receptive field of the deployed model variant:
#   standard TCN (kernel=7, 5 layers): 187
#   small TCN    (kernel=5, 4 layers):  61
#   large TCN    (kernel=9, 6 layers): 505
WINDOW_SIZE = 187
FEATURE_COUNT = 28  # 24 IMU + 4 joint angles
OUTPUT_COUNT = 4    # hip_l, hip_r, knee_l, knee_r  (Nm/kg)

# ---------------------------------------------------------------------------
# App lifespan: load model once at startup
# ---------------------------------------------------------------------------

_session: ort.InferenceSession | None = None
_input_name: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _session, _input_name

    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        raise RuntimeError(f"Model file not found: {model_path.resolve()}")

    print(f"Loading ONNX model: {model_path.resolve()}")
    _session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    _input_name = _session.get_inputs()[0].name
    print(f"Model loaded. Input: '{_input_name}' | Expected shape: (1, {WINDOW_SIZE}, {FEATURE_COUNT})")

    yield

    _session = None
    _input_name = None
    print("Server shutting down.")


app = FastAPI(
    title="Exoskeleton TCN Inference Server",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    # List of WINDOW_SIZE rows, each with FEATURE_COUNT values.
    # C++ sends: {"window": [[f0..f27], [f0..f27], ...]}  (187 rows)
    window: Annotated[
        list[list[float]],
        Field(description=f"Sensor window: {WINDOW_SIZE} timesteps × {FEATURE_COUNT} features"),
    ]

    @model_validator(mode="after")
    def validate_shape(self) -> "PredictRequest":
        rows = len(self.window)
        if rows != WINDOW_SIZE:
            raise ValueError(f"Expected {WINDOW_SIZE} timesteps, got {rows}")
        for i, row in enumerate(self.window):
            if len(row) != FEATURE_COUNT:
                raise ValueError(
                    f"Timestep {i}: expected {FEATURE_COUNT} features, got {len(row)}"
                )
        return self


class PredictResponse(BaseModel):
    # Torque predictions for the 4 joints at the latest timestep (Nm/kg)
    hip_left: float
    hip_right: float
    knee_left: float
    knee_right: float
    inference_ms: float  # ONNX session.run() time only, excludes HTTP/JSON overhead


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def _run_inference(window_bytes: np.ndarray) -> tuple[np.ndarray, float]:
    """Shared inference logic. Input must already be shaped (1, WINDOW_SIZE, FEATURE_COUNT)."""
    t0 = time.perf_counter()
    outputs = _session.run(None, {_input_name: window_bytes})  # type: ignore[index]
    inference_ms = (time.perf_counter() - t0) * 1000
    preds = outputs[0][0, -1, :]
    return preds, inference_ms


@app.get("/health")
def health() -> dict[str, str]:
    """Quick liveness check — the C++ side can poll this on startup."""
    if _session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if _session is None or _input_name is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    input_array = np.array(req.window, dtype=np.float32)[np.newaxis, ...]
    preds, inference_ms = _run_inference(input_array)

    return PredictResponse(
        hip_left=float(preds[0]),
        hip_right=float(preds[1]),
        knee_left=float(preds[2]),
        knee_right=float(preds[3]),
        inference_ms=round(inference_ms, 3),
    )


@app.post("/predict_msgpack")
async def predict_msgpack(request: Request) -> Response:
    """
    Msgpack endpoint for lower-overhead inference.

    Request body:  msgpack-encoded flat list of 187*28=5236 float32 values (row-major)
    Response body: msgpack-encoded dict {hip_left, hip_right, knee_left, knee_right, inference_ms}
    """
    if _session is None or _input_name is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    raw = await request.body()
    flat = msgpack.unpackb(raw, raw=False)
    input_array = np.array(flat, dtype=np.float32).reshape(1, WINDOW_SIZE, FEATURE_COUNT)
    preds, inference_ms = _run_inference(input_array)

    result = {
        "hip_left": float(preds[0]),
        "hip_right": float(preds[1]),
        "knee_left": float(preds[2]),
        "knee_right": float(preds[3]),
        "inference_ms": round(inference_ms, 3),
    }
    return Response(content=msgpack.packb(result), media_type="application/x-msgpack")
