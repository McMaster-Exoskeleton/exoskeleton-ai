"""
Quick server test script for Raspberry Pi.
Tests the inference server with dummy data and reports latency.

Usage:
    python scripts/test_server.py
    python scripts/test_server.py --port 8000 --runs 20
"""

import argparse
import json
import time
import urllib.request
import urllib.error

import msgpack
import numpy as np

DEFAULT_HOST = "http://127.0.0.1"
DEFAULT_PORT = 8000
WINDOW_SIZE = 187
FEATURE_COUNT = 28


def check_health(base_url: str) -> bool:
    try:
        resp = urllib.request.urlopen(f"{base_url}/health", timeout=3)
        data = json.loads(resp.read())
        return data.get("status") == "ok"
    except Exception as e:
        print(f"  Health check failed: {e}")
        return False


def predict(base_url: str, window: list) -> dict:
    body = json.dumps({"window": window}).encode()
    req = urllib.request.Request(
        f"{base_url}/predict",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=5)
    return json.loads(resp.read())


def predict_msgpack(base_url: str, flat: list) -> dict:
    # Send flat list of 187*28 floats, receive msgpack dict
    body = msgpack.packb(flat)
    req = urllib.request.Request(
        f"{base_url}/predict_msgpack",
        data=body,
        headers={"Content-Type": "application/x-msgpack"},
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=5)
    return msgpack.unpackb(resp.read(), raw=False)


def main():
    parser = argparse.ArgumentParser(description="Test exoskeleton inference server")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--runs", type=int, default=20, help="Number of latency runs")
    args = parser.parse_args()

    base_url = f"{args.host}:{args.port}"

    print("=" * 50)
    print("Exoskeleton Server Test")
    print("=" * 50)
    print(f"Server: {base_url}")

    # Health check
    print("\n[1] Health check...")
    if not check_health(base_url):
        print("Server not ready. Is it running?")
        print(f"  uvicorn scripts.server:app --host 127.0.0.1 --port {args.port}")
        return
    print("  OK")

    # Single prediction with zeros
    print("\n[2] Single prediction (zero input)...")
    window = np.zeros((WINDOW_SIZE, FEATURE_COUNT)).tolist()
    result = predict(base_url, window)
    print(f"  hip_left:   {result['hip_left']:.6f} Nm/kg")
    print(f"  hip_right:  {result['hip_right']:.6f} Nm/kg")
    print(f"  knee_left:  {result['knee_left']:.6f} Nm/kg")
    print(f"  knee_right: {result['knee_right']:.6f} Nm/kg")

    # Single prediction with random data
    print("\n[3] Single prediction (random input)...")
    window = np.random.randn(WINDOW_SIZE, FEATURE_COUNT).tolist()
    result = predict(base_url, window)
    print(f"  hip_left:   {result['hip_left']:.6f} Nm/kg")
    print(f"  hip_right:  {result['hip_right']:.6f} Nm/kg")
    print(f"  knee_left:  {result['knee_left']:.6f} Nm/kg")
    print(f"  knee_right: {result['knee_right']:.6f} Nm/kg")

    # Latency benchmark — JSON
    print(f"\n[4] Latency benchmark — JSON ({args.runs} runs)...")
    window = np.random.randn(WINDOW_SIZE, FEATURE_COUNT).tolist()
    roundtrip_times = []
    inference_times = []
    for i in range(args.runs):
        t0 = time.perf_counter()
        result = predict(base_url, window)
        roundtrip_times.append((time.perf_counter() - t0) * 1000)
        inference_times.append(result["inference_ms"])
        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{args.runs}")

    roundtrip_times = np.array(roundtrip_times)
    inference_times = np.array(inference_times)
    overhead = roundtrip_times - inference_times

    print(f"\n  {'':30s} {'round-trip':>10s}  {'inference':>10s}  {'overhead':>10s}")
    print(f"  {'':30s} {'(HTTP+JSON)':>10s}  {'(ONNX only)':>10s}  {'(diff)':>10s}")
    print(f"  {'-'*64}")
    print(f"  {'mean':30s} {roundtrip_times.mean():>10.1f}  {inference_times.mean():>10.1f}  {overhead.mean():>10.1f}  ms")
    print(f"  {'median':30s} {np.median(roundtrip_times):>10.1f}  {np.median(inference_times):>10.1f}  {np.median(overhead):>10.1f}  ms")
    print(f"  {'min':30s} {roundtrip_times.min():>10.1f}  {inference_times.min():>10.1f}  {overhead.min():>10.1f}  ms")
    print(f"  {'max':30s} {roundtrip_times.max():>10.1f}  {inference_times.max():>10.1f}  {overhead.max():>10.1f}  ms")
    print(f"  {'p95':30s} {np.percentile(roundtrip_times, 95):>10.1f}  {np.percentile(inference_times, 95):>10.1f}  {np.percentile(overhead, 95):>10.1f}  ms")

    # Latency benchmark — msgpack
    print(f"\n[5] Latency benchmark — msgpack ({args.runs} runs)...")
    flat = np.random.randn(WINDOW_SIZE, FEATURE_COUNT).flatten().tolist()
    mp_roundtrip_times = []
    mp_inference_times = []
    for i in range(args.runs):
        t0 = time.perf_counter()
        result = predict_msgpack(base_url, flat)
        mp_roundtrip_times.append((time.perf_counter() - t0) * 1000)
        mp_inference_times.append(result["inference_ms"])
        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{args.runs}")

    mp_roundtrip_times = np.array(mp_roundtrip_times)
    mp_inference_times = np.array(mp_inference_times)
    mp_overhead = mp_roundtrip_times - mp_inference_times

    budget_ms = 10.0  # 100 Hz

    print(f"\n  {'':20s} {'JSON rt':>8s}  {'MP rt':>8s}  {'inference':>10s}  {'JSON oh':>8s}  {'MP oh':>8s}")
    print(f"  {'-'*70}")

    def row(label, json_rt, mp_rt, inf, json_oh, mp_oh):
        print(f"  {label:20s} {json_rt:>8.1f}  {mp_rt:>8.1f}  {inf:>10.1f}  {json_oh:>8.1f}  {mp_oh:>8.1f}  ms")

    row("mean",   roundtrip_times.mean(),           mp_roundtrip_times.mean(),           mp_inference_times.mean(),           overhead.mean(),           mp_overhead.mean())
    row("median", np.median(roundtrip_times),        np.median(mp_roundtrip_times),        np.median(mp_inference_times),        np.median(overhead),        np.median(mp_overhead))
    row("min",    roundtrip_times.min(),             mp_roundtrip_times.min(),             mp_inference_times.min(),             overhead.min(),             mp_overhead.min())
    row("max",    roundtrip_times.max(),             mp_roundtrip_times.max(),             mp_inference_times.max(),             overhead.max(),             mp_overhead.max())
    row("p95",    np.percentile(roundtrip_times,95), np.percentile(mp_roundtrip_times,95), np.percentile(mp_inference_times,95), np.percentile(overhead,95), np.percentile(mp_overhead,95))

    print(f"\n  Control loop budget (100 Hz): {budget_ms:.0f} ms")
    for label, times in [("JSON", roundtrip_times), ("msgpack", mp_roundtrip_times)]:
        if times.mean() < budget_ms:
            print(f"  {label}: PASS")
        else:
            print(f"  {label}: WARN — exceeds budget")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
