# ML–Control Interface Specification

> **Version**: 1.0.0  
> **Status**: Draft — Pending Embedded Team Review  
> **Last Updated**: 2026-01-21

---

## Overview

This document defines the **interface contract** between the Machine Learning (Movement Prediction) module and the Control/Embedded system for the McMaster Exoskeleton project.

**Purpose**: Enable independent development by both teams while ensuring integration compatibility.

---

## Input Specification

The ML model receives a sequence of sensor measurements collected by the embedded system.

### Tensor Shape

```
(Batch_Size, Sequence_Length, 28)
```

| Dimension | Symbol | Description |
|-----------|--------|-------------|
| Batch Size | `B` | Number of parallel inference streams. **Real-time: 1** |
| Sequence Length | `L` | Variable length history (minimum ~100 samples recommended) |
| Input Features | `C_in` | **28** features (24 IMU + 4 encoder angles) |

### Data Type

- **dtype**: `float32`

### Feature Ordering

The 28 input features **must** be provided in the following order:

#### IMU Features (Indices 0–23)

| Index | Feature Name | Unit | Description |
|-------|--------------|------|-------------|
| 0 | `thigh_imu_l_accel_x` | m/s² | Left thigh IMU — Acceleration X |
| 1 | `thigh_imu_l_accel_y` | m/s² | Left thigh IMU — Acceleration Y |
| 2 | `thigh_imu_l_accel_z` | m/s² | Left thigh IMU — Acceleration Z |
| 3 | `thigh_imu_l_gyro_x` | rad/s | Left thigh IMU — Gyroscope X |
| 4 | `thigh_imu_l_gyro_y` | rad/s | Left thigh IMU — Gyroscope Y |
| 5 | `thigh_imu_l_gyro_z` | rad/s | Left thigh IMU — Gyroscope Z |
| 6 | `shank_imu_l_accel_x` | m/s² | Left shank IMU — Acceleration X |
| 7 | `shank_imu_l_accel_y` | m/s² | Left shank IMU — Acceleration Y |
| 8 | `shank_imu_l_accel_z` | m/s² | Left shank IMU — Acceleration Z |
| 9 | `shank_imu_l_gyro_x` | rad/s | Left shank IMU — Gyroscope X |
| 10 | `shank_imu_l_gyro_y` | rad/s | Left shank IMU — Gyroscope Y |
| 11 | `shank_imu_l_gyro_z` | rad/s | Left shank IMU — Gyroscope Z |
| 12 | `thigh_imu_r_accel_x` | m/s² | Right thigh IMU — Acceleration X |
| 13 | `thigh_imu_r_accel_y` | m/s² | Right thigh IMU — Acceleration Y |
| 14 | `thigh_imu_r_accel_z` | m/s² | Right thigh IMU — Acceleration Z |
| 15 | `thigh_imu_r_gyro_x` | rad/s | Right thigh IMU — Gyroscope X |
| 16 | `thigh_imu_r_gyro_y` | rad/s | Right thigh IMU — Gyroscope Y |
| 17 | `thigh_imu_r_gyro_z` | rad/s | Right thigh IMU — Gyroscope Z |
| 18 | `shank_imu_r_accel_x` | m/s² | Right shank IMU — Acceleration X |
| 19 | `shank_imu_r_accel_y` | m/s² | Right shank IMU — Acceleration Y |
| 20 | `shank_imu_r_accel_z` | m/s² | Right shank IMU — Acceleration Z |
| 21 | `shank_imu_r_gyro_x` | rad/s | Right shank IMU — Gyroscope X |
| 22 | `shank_imu_r_gyro_y` | rad/s | Right shank IMU — Gyroscope Y |
| 23 | `shank_imu_r_gyro_z` | rad/s | Right shank IMU — Gyroscope Z |

#### Encoder Angle Features (Indices 24–27)

| Index | Feature Name | Unit | Description |
|-------|--------------|------|-------------|
| 24 | `hip_flexion_l` | rad | Left hip encoder angle |
| 25 | `knee_angle_l` | rad | Left knee encoder angle |
| 26 | `hip_flexion_r` | rad | Right hip encoder angle |
| 27 | `knee_angle_r` | rad | Right knee encoder angle |

---

## Output Specification

The ML model outputs predicted biological joint moments.

### Tensor Shape

```
(Batch_Size, Sequence_Length, 4)
```

For real-time control, use only the **last timestep**: `output[:, -1, :]` → shape `(1, 4)`

### Data Type

- **dtype**: `float32`

### Feature Ordering

| Index | Feature Name | Unit | Description |
|-------|--------------|------|-------------|
| 0 | `hip_moment_l` | Nm/kg | Left hip biological joint moment |
| 1 | `knee_moment_l` | Nm/kg | Left knee biological joint moment |
| 2 | `hip_moment_r` | Nm/kg | Right hip biological joint moment |
| 3 | `knee_moment_r` | Nm/kg | Right knee biological joint moment |

> [!IMPORTANT]
> Output moments are **normalized by body mass** (Nm/kg). To obtain absolute torque (Nm), the embedded controller must multiply by the participant's mass in kg:
> ```
> torque_nm = moment_nm_per_kg * participant_mass_kg
> ```

---

## Timing & Real-Time Constraints

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Sampling Rate** | 100 Hz | Sensor data collected every 10 ms |
| **Inference Latency** | < 5 ms | Target; allows margin for control loop overhead |
| **Model Causality** | Strictly Causal | Prediction at time *t* uses only inputs from *t* and earlier |
| **Minimum History** | ~100 samples | ~1 second of context for TCN receptive field |

### Control Loop Timing Diagram

```
Time (ms):  0      10      20      30      40
            │       │       │       │       │
Sensor:     ├───┬───┼───┬───┼───┬───┼───┬───┤
            │ S₀│   │ S₁│   │ S₂│   │ S₃│   │
            └───┘   └───┘   └───┘   └───┘   
                    ↓
              ┌─────────────┐
              │  Inference  │ < 5ms
              └──────┬──────┘
                     ↓
              ┌─────────────┐
              │  Actuation  │
              └─────────────┘
```

---

## Variable Sequence Length

The TCN (Temporal Convolutional Network) architecture is a **Fully Convolutional Network**, which natively supports variable-length input sequences.

### Training
- Sequences are padded to batch-max length
- A boolean mask indicates valid (non-padded) timesteps
- Loss is computed only on valid timesteps

### Inference (Real-Time)
- Maintain a rolling buffer of recent samples
- No fixed window size required
- Recommended minimum buffer: 100 samples (~1 second)

---

## Example Inference Call

See [example_inference.py](./example_inference.py) for a complete working example.

### Minimal Code Snippet

```python
import torch

# Load trained model
model = torch.load("models/tcn_moment_predictor.pt")
model.eval()

# Prepare input: (1, sequence_length, 28)
sensor_buffer = torch.tensor(sensor_data, dtype=torch.float32).unsqueeze(0)

# Run inference
with torch.no_grad():
    output = model(sensor_buffer)  # (1, L, 4)
    current_moment = output[:, -1, :]  # (1, 4) — last timestep only

# Denormalize to absolute torque
torque_nm = current_moment * participant_mass_kg
```

---

## Normalization

### Input Normalization

    The model expects **normalized inputs**. Normalization statistics are computed from the training set and stored in:

    ```
    data/processed/phase1/normalization_stats.json
    ```

    Formula:
    ```
    x_normalized = (x_raw - mean) / (std + 1e-8)
    ```

> [!NOTE]
> The embedded system should either:
> 1. Apply normalization before sending data to the ML module, OR
> 2. Send raw data and let the ML module handle normalization

### Output Denormalization

If the model was trained with normalized targets, apply inverse transform:
```
y_raw = y_normalized * std + mean
```

---

## Error Handling

| Error Condition | Recommended Action |
|-----------------|-------------------|
| Input shape mismatch | Return error code; do not actuate |
| NaN/Inf in input | Clamp or skip inference cycle |
| Inference timeout (> 10ms) | Use previous prediction; log warning |
| Model load failure | Fall back to safe mode (zero torque) |

---

## Versioning

This interface follows semantic versioning:

- **MAJOR**: Breaking changes to tensor shapes or feature ordering
- **MINOR**: Additions (new optional features)
- **PATCH**: Documentation or implementation fixes

Current version: **1.0.0**

---

## Open Questions

> [!WARNING]
> The following items require confirmation from the controls/embedded team:

1. **Sampling Rate**: Is 100 Hz confirmed, or could it change?
2. **Latency Budget**: Is < 5 ms acceptable?
3. **Denormalization Responsibility**: Should the ML module output absolute torque (Nm), or is Nm/kg acceptable?
4. **Joint Order**: Confirm `[hip_l, knee_l, hip_r, knee_r]` matches actuator wiring.
5. **IMU Frame Convention**: Confirm axis orientation (NED? ENU?).

---

## References

- [Data Infrastructure Plan](./data_infrastructure_plan.md)
- [ExoskeletonDataset Class](./src/exoskeleton_ml/data/datasets.py)
