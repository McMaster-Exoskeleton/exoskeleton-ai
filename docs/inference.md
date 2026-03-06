# Inference System — Integration Guide

> **Audience**: Embedded / firmware team
> **Last Updated**: 2026-03-06
> **Related**: [ML–Control Interface Specification](./ml_control_interface.md)

This document covers everything needed to get torque predictions from the TCN model at runtime. It does not repeat the tensor spec or feature ordering — those are fully defined in [`ml_control_interface.md`](./ml_control_interface.md). Read that first.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Deployment Modes](#deployment-modes)
4. [Mode A: HTTP Server](#mode-a-http-server)
   - [Starting the Server](#starting-the-server)
   - [Endpoint Reference](#endpoint-reference)
   - [POST /predict (JSON)](#post-predict-json)
   - [POST /predict_msgpack (Binary)](#post-predict_msgpack-binary)
   - [GET /health](#get-health)
   - [C++ Integration Example](#c-integration-example)
5. [Mode B: Serial (Direct Arduino)](#mode-b-serial-direct-arduino)
   - [Protocol](#protocol)
   - [Running the Script](#running-the-script)
6. [Choosing a Mode](#choosing-a-mode)
7. [Model Export](#model-export)
8. [Validation & Testing](#validation--testing)
9. [Error Handling](#error-handling)
10. [Configuration Reference](#configuration-reference)

---

## Overview

The ML system runs on a **Raspberry Pi** and exposes the trained TCN model for real-time torque prediction. At each control cycle, your system sends a window of sensor readings and receives 4 joint torque values:

| Output | Unit | Description |
|--------|------|-------------|
| `hip_left` | Nm/kg | Left hip biological joint moment |
| `hip_right` | Nm/kg | Right hip biological joint moment |
| `knee_left` | Nm/kg | Left knee biological joint moment |
| `knee_right` | Nm/kg | Right knee biological joint moment |

> **Important**: Outputs are normalized by body mass. Multiply by participant mass (kg) to get absolute torque in Nm.
> See [ml_control_interface.md — Output Specification](./ml_control_interface.md#output-specification) for full details.

There is one production-ready deployment mode and one reference implementation:

| Mode | Script | Interface | Status |
|------|--------|-----------|--------|
| **A — HTTP Server** | `scripts/server.py` | HTTP POST (JSON or msgpack) | Production-ready |
| **B — Serial** | `scripts/endpoint_tcn.py` | UART serial | Reference implementation only |

---

## Prerequisites

### Hardware
- Raspberry Pi (tested on Pi 4 / Pi 5)
- ONNX model file at `outputs/model.onnx` on the Pi (see [Model Export](#model-export))

### Python dependencies (install on the Pi)

```bash
pip install fastapi "uvicorn[standard]" onnxruntime numpy msgpack
```

For serial mode only:

```bash
pip install pyserial
```

For C++ test client only:

```bash
sudo apt install libcurl4-openssl-dev nlohmann-json3-dev libmsgpack-dev
```

---

## Deployment Modes

### Mode A: HTTP Server

The server loads the ONNX model once at startup and serves predictions over HTTP. The C++ control system calls `POST /predict` or `POST /predict_msgpack` on each control cycle.

```
C++ control loop
      │
      │  POST /predict_msgpack  (recommended)
      │  POST /predict          (JSON fallback)
      ▼
FastAPI server (scripts/server.py)
      │
      ▼
ONNX Runtime (outputs/model.onnx)
      │
      ▼
{ hip_left, hip_right, knee_left, knee_right, inference_ms }
```

#### Starting the Server

```bash
# From the exoskeleton-ai repo root on the Pi:

# Accessible only from localhost (recommended for same-Pi integration):
uvicorn scripts.server:app --host 127.0.0.1 --port 8000

# Accessible over the network (e.g. laptop → Pi):
uvicorn scripts.server:app --host 0.0.0.0 --port 8000
```

The server prints confirmation when the model is loaded:

```
Loading ONNX model: /home/pi/exoskeleton-ai/outputs/model.onnx
Model loaded. Input: 'input' | Expected shape: (1, 187, 28)
```

---

#### Endpoint Reference

##### `POST /predict` (JSON)

Standard JSON request/response. Use this for debugging or if your C++ environment does not have a msgpack library.

**Request**

- Content-Type: `application/json`
- Body: JSON object with a single key `window` — a 2D array of shape `[WINDOW_SIZE][28]`

```json
{
  "window": [
    [f0, f1, ..., f27],
    [f0, f1, ..., f27],
    ...
  ]
}
```

`window` must have exactly `WINDOW_SIZE` rows (187 for the standard TCN). Each row must have exactly 28 values in the feature order defined in [`ml_control_interface.md`](./ml_control_interface.md#feature-ordering).

**Response**

- Content-Type: `application/json`
- HTTP 200 on success, 422 on shape validation error, 503 if model not loaded

```json
{
  "hip_left":    0.1234,
  "hip_right":   0.1102,
  "knee_left":  -0.0873,
  "knee_right": -0.0941,
  "inference_ms": 3.241
}
```

| Field | Type | Description |
|-------|------|-------------|
| `hip_left` | float | Left hip moment (Nm/kg) |
| `hip_right` | float | Right hip moment (Nm/kg) |
| `knee_left` | float | Left knee moment (Nm/kg) |
| `knee_right` | float | Right knee moment (Nm/kg) |
| `inference_ms` | float | ONNX session time only — does not include HTTP or JSON overhead |

---

##### `POST /predict_msgpack` (Binary)

Preferred endpoint for the C++ control loop. Uses msgpack binary encoding — lower serialization overhead than JSON, especially important at 100 Hz.

**Request**

- Content-Type: `application/x-msgpack`
- Body: msgpack-encoded flat array of `WINDOW_SIZE * 28 = 5236` float32 values, **row-major order**
  - i.e. timestep 0's 28 features, then timestep 1's 28 features, ..., then timestep 186's 28 features

**Response**

- Content-Type: `application/x-msgpack`
- Body: msgpack-encoded map with the same 5 keys as the JSON response

```
{
  "hip_left":    <float>,
  "hip_right":   <float>,
  "knee_left":   <float>,
  "knee_right":  <float>,
  "inference_ms": <float>
}
```

**C++ Integration Example**

```cpp
#include <curl/curl.h>
#include <msgpack.hpp>
#include <vector>
#include <map>
#include <string>

// Build request body: flat float32 array, row-major, shape (187, 28)
std::string build_msgpack_request(const std::vector<float>& flat) {
    msgpack::sbuffer buf;
    msgpack::pack(buf, flat);
    return std::string(buf.data(), buf.size());
}

// Parse response body into prediction values
struct Prediction {
    double hip_left, hip_right, knee_left, knee_right, inference_ms;
};

Prediction parse_msgpack_response(const std::string& body) {
    msgpack::object_handle oh = msgpack::unpack(body.data(), body.size());
    std::map<std::string, double> result;
    oh.get().convert(result);
    return {
        result["hip_left"],
        result["hip_right"],
        result["knee_left"],
        result["knee_right"],
        result["inference_ms"],
    };
}

// In your control loop:
// 1. Fill flat[] with your (187 * 28) sensor window, row-major
// 2. POST to http://127.0.0.1:8000/predict_msgpack
// 3. Parse response and use prediction values
```

For a complete working example including libcurl boilerplate, health check, and latency benchmarking, see `scripts/test_endpoint.cpp`.

---

##### `GET /health`

Liveness check. Returns HTTP 200 when the model is loaded and ready, HTTP 503 otherwise. Poll this on startup before sending predictions.

**Response (200)**
```json
{ "status": "ok" }
```

---

### Mode B: Serial (Reference Implementation)

> **This is not a production-ready script.** `scripts/endpoint_tcn.py` is a proof-of-concept demonstrating how a serial inference loop would work once an Arduino with IMU hardware is available. It is not suitable for deployment as-is — it is missing argument parsing, graceful shutdown, and runtime configuration. Use it as a starting point when the embedded hardware is ready.

The script runs on the Pi and communicates with an Arduino over UART (`/dev/ttyACM0`, 115200 baud). It reads one sample at a time from serial, maintains a rolling window, runs ONNX inference, and writes 4 torque values back.

#### Intended Protocol

**Arduino → Pi (input)**

Send one line of comma-separated floats over serial at each sample:

```
f0,f1,f2,...,f27\n
```

- 28 values per line, matching the feature order in [`ml_control_interface.md`](./ml_control_interface.md#feature-ordering)
- Baud rate: **115200**
- One sample per line, terminated with `\n`

The Pi maintains a rolling window of the last 100 samples internally. No need to send the full window — just one sample at a time.

**Pi → Arduino (output)**

After each inference, the Pi sends back one line:

```
hip_left,hip_right,knee_left,knee_right\n
```

- 4 comma-separated floats, 4 decimal places each
- Units: Nm/kg (multiply by body mass for absolute torque)
- Example: `0.1234,0.1102,-0.0873,-0.0941\n`

#### Known Limitations

- Serial port and baud rate are hardcoded — cannot be overridden at runtime without editing the file
- `argparse` is imported but not wired up
- No graceful shutdown (Ctrl+C will leave the serial port open)
- Window size (100) does not match the HTTP server (187) — accuracy implications TBD with the ML team
- Docstring is incomplete

---

## Choosing a Mode

**Use Mode A (HTTP Server)** for any current integration work. It is the only production-ready option.

**Mode B (Serial)** describes the intended protocol for direct Arduino ↔ Pi communication but requires further development before it can be deployed. Refer to it when building out the embedded serial interface.

| Consideration | HTTP Server (Mode A) | Serial (Mode B) |
|--------------|----------------------|-----------------|
| Status | Production-ready | Reference implementation |
| Interface | HTTP POST | UART |
| Overhead | Low (msgpack) / Moderate (JSON) | Minimal |
| Latency at 100 Hz | Typically passes 10 ms budget (verify with benchmark) | Lowest possible |
| Architecture | Pi runs server; C++ calls it | Pi reads/writes serial directly |
| Debugging | Easy — curl / Postman / browser | Requires serial monitor |

---

## Model Export

The deployed model must be in ONNX format (`outputs/model.onnx`). If the ML team has given you a `.onnx` file, copy it to the Pi:

```bash
scp outputs/model.onnx pi@<PI_IP>:~/exoskeleton-ai/outputs/model.onnx
```

If you need to regenerate the ONNX file from a trained PyTorch checkpoint:

```bash
# Default path (outputs/2026-01-08/18-29-05/best_model.pt):
python scripts/export_onnx.py

# Specify a checkpoint:
python scripts/export_onnx.py --model-path outputs/<date>/<time>/best_model.pt

# Specify output path:
python scripts/export_onnx.py --output outputs/model.onnx
```

The script verifies that ONNX and PyTorch outputs match within tolerance before saving.

---

## Validation & Testing

### Python test client

Tests the HTTP server with dummy data and reports latency statistics:

```bash
# Basic test (server must be running):
python scripts/test_server.py

# Custom host/port and more runs:
python scripts/test_server.py --host http://127.0.0.1 --port 8000 --runs 50
```

Output includes mean, median, min, max, p95 for round-trip time, ONNX-only inference time, and HTTP overhead — separately for JSON and msgpack endpoints. Reports PASS/WARN against the 10 ms (100 Hz) budget.

### C++ test client

Tests both endpoints from C++, matching the actual integration environment:

```bash
# Compile (on the Pi):
g++ -O2 -std=c++17 scripts/test_endpoint.cpp -lcurl -o test_endpoint

# Run (server must be running):
./test_endpoint
./test_endpoint --runs 50
```

Produces the same latency table as the Python client.

### Quick curl check

```bash
# Health check:
curl http://127.0.0.1:8000/health

# Single prediction with Python-generated data:
python -c "
import json, numpy as np
window = np.zeros((187, 28)).tolist()
print(json.dumps({'window': window}))
" | curl -s -X POST http://127.0.0.1:8000/predict \
     -H 'Content-Type: application/json' \
     -d @- | python -m json.tool
```

---

## Error Handling

| HTTP Status | Meaning | Action |
|-------------|---------|--------|
| 200 | Success | Use prediction values |
| 422 | Invalid input shape | Check window dimensions — must be exactly `(WINDOW_SIZE, 28)` |
| 503 | Model not loaded | Server is starting up or crashed — poll `/health` and retry |
| Timeout / no response | Server not running | Check that `uvicorn` is running on the Pi |

General guidance from the interface spec:

| Condition | Recommended Action |
|-----------|-------------------|
| Inference timeout (> 10 ms) | Use previous prediction; log warning |
| NaN / Inf in input | Clamp or skip cycle |
| Server unreachable at startup | Wait and retry; abort if unresolved |
| Model load failure | Fall back to safe mode (zero torque) |

---

## Configuration Reference

All constants that may need adjustment are at the top of each script.

### `scripts/server.py`

| Constant | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `outputs/model.onnx` | ONNX model file path |
| `WINDOW_SIZE` | `187` | Timesteps per request (must match exported model) |
| `FEATURE_COUNT` | `28` | Input features per timestep — do not change |
| `OUTPUT_COUNT` | `4` | Output joints — do not change |

> TCN window size variants: standard (kernel=7, 5 layers) = **187**, small (kernel=5, 4 layers) = **61**, large (kernel=9, 6 layers) = **505**. Confirm with ML team which variant is deployed.

### `scripts/endpoint_tcn.py`

| Constant | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `outputs/model.onnx` | ONNX model file path |
| `SERIAL_PORT` | `/dev/ttyACM0` | Arduino serial port |
| `BAUD_RATE` | `115200` | Serial baud rate |
| `FEATURE_COUNT` | `28` | Input features — do not change |
| `WINDOW_SIZE` | `100` | Rolling buffer length |

### `scripts/test_endpoint.cpp`

| Constant | Default | Description |
|----------|---------|-------------|
| `HEALTH_URL` | `http://127.0.0.1:8000/health` | Health endpoint URL |
| `JSON_URL` | `http://127.0.0.1:8000/predict` | JSON endpoint URL |
| `MSGPACK_URL` | `http://127.0.0.1:8000/predict_msgpack` | Msgpack endpoint URL |
| `WINDOW_SIZE` | `187` | Must match server |
| `FEATURE_COUNT` | `28` | Must match server |
| `BUDGET_MS` | `10.0` | Latency budget for PASS/WARN check |

---

*For questions about the model itself, training, or the feature specification, contact the ML team or refer to [`ml_control_interface.md`](./ml_control_interface.md).*
