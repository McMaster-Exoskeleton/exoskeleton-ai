"""
ONNX-only inference script for Raspberry Pi deployment. This script is for using

"""

import onnxruntime
import numpy as np
import argparse
from pathlib import Path
import serial
from collections import deque

MODEL_PATH = "outputs/model.onnx"
SERIAL_PORT = "/dev/ttyACM0" 
BAUD_RATE = 115200
FEATURE_COUNT = 28
WINDOW_SIZE = 100

def main():
    

    if not Path(MODEL_PATH).exists():
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        return
    
    print(f"Loading model: {MODEL_PATH}")

    session = onnxruntime.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    history = deque(maxlen=WINDOW_SIZE)
    for _ in range(WINDOW_SIZE):
        history.append(np.zeros(FEATURE_COUNT))

    try: 
        ser = serial.Serial(SERIAL_PORT, 115200, timeout=1)
        ser.flush()
    except Exception as e:
        print(f"Serial Error: {e}")
        return

    # 3. Run Inference

    while True:
        if ser.in_waiting > 0:
            try:
                line = ser.readline().decode('utf-8').strip()
                if not line: continue
                
                # Parse the latest single reading from Arduino
                new_sample = [float(x) for x in line.split(',')]
                if len(new_sample) != FEATURE_COUNT: continue

                # 2. Add new sample to history (automatically drops the oldest)
                history.append(new_sample)

                # 3. Convert history to the shape the TCN expects: (1, 100, 28)
                input_data = np.array(history, dtype=np.float32).reshape(1, WINDOW_SIZE, FEATURE_COUNT)
            
                # 4. Run Inference
                outputs = session.run(None, {input_name: input_data})
                predictions = outputs[0] # Shape: (1, 100, 4)

                # 5. Extract the LATEST prediction for the 4 motors
                # We take index -1 (the most recent time step)
                current_pred = predictions[0, -1, :] 

                # 6. Send all 4 motor values back as a comma-separated string
                pred_str = ",".join([f"{x:.4f}" for x in current_pred])
                ser.write(f"{pred_str}\n".encode('utf-8'))

            except Exception as e:
                   print(f"Error: {e}")

if __name__ == "__main__":
    main()


        