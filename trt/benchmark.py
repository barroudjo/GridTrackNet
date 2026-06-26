"""Benchmark TensorRT FP16 vs the TensorFlow/Keras baseline, and check accuracy.

Reports per-instance latency / FPS for the model forward pass only (one instance
= 5 frames). Also reports how often the TensorRT ball-detection decision (grid
argmax + 0.5 threshold) matches Keras, which is what actually affects Predict.py.
"""
import argparse
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from trt.trt_runner import TRTRunner  # noqa: E402

CHANNELS, HEIGHT, WIDTH = 15, 432, 768


def timeit(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def decisions(y):
    """Replicate Predict.py: per (instance, frame) -> (has_ball, row, col)."""
    conf = y[:, 0::3, :, :]  # (N,5,27,48) after the channel layout used in Predict
    n, f = conf.shape[0], conf.shape[1]
    out = []
    for i in range(n):
        for j in range(f):
            g = conf[i, j]
            out.append((g.max() >= 0.5, *np.unravel_index(np.argmax(g), g.shape)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=os.path.join(REPO_ROOT, "trt", "gridtracknet_fp16.trt"))
    ap.add_argument("--batch", type=int, default=1, help="instances per forward pass")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--skip-tf", action="store_true", help="skip the slow TF-CPU baseline")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    x = rng.random((args.batch, CHANNELS, HEIGHT, WIDTH), dtype=np.float32)

    # --- TensorRT FP16 ---
    runner = TRTRunner(args.engine)
    trt_out = runner.infer(x)
    trt_lat = timeit(lambda: runner.infer(x), args.iters, args.warmup)
    fps_per_frame = args.batch * 5 / trt_lat  # 5 frames per instance
    print("=== TensorRT FP16 (RTX 3050) ===")
    print(f"batch (instances): {args.batch}")
    print(f"latency/forward  : {trt_lat * 1e3:.3f} ms")
    print(f"throughput       : {args.batch / trt_lat:.1f} instances/s  |  {fps_per_frame:.1f} frames/s")

    if args.skip_tf:
        return

    # --- TF/Keras baseline (CPU here, since GPU libs for TF 2.11 aren't installed) ---
    import tensorflow as tf  # noqa
    from GridTrackNet import GridTrackNet

    model = GridTrackNet(5, HEIGHT, WIDTH)
    model.load_weights(os.path.join(REPO_ROOT, "model_weights.h5"))
    keras_out = model.predict(x, verbose=0)
    tf_lat = timeit(lambda: model.predict(x, verbose=0), max(args.iters // 10, 5), 3)
    print("\n=== TensorFlow/Keras baseline (CPU) ===")
    print(f"latency/forward  : {tf_lat * 1e3:.3f} ms")
    print(f"throughput       : {args.batch / tf_lat:.1f} instances/s  |  {args.batch * 5 / tf_lat:.1f} frames/s")

    print(f"\n>>> TensorRT speedup vs TF-CPU: {tf_lat / trt_lat:.1f}x")

    # --- Accuracy: raw tensor diff + detection-decision agreement ---
    max_abs = float(np.max(np.abs(keras_out - trt_out)))
    kd, td = decisions(keras_out), decisions(trt_out)
    agree = sum(a == b for a, b in zip(kd, td)) / len(kd)
    print("\n=== Accuracy (FP16 TRT vs FP32 Keras) ===")
    print(f"max abs diff on raw outputs : {max_abs:.3e}")
    print(f"detection-decision agreement: {agree * 100:.1f}%  ({len(kd)} frame-decisions)")


if __name__ == "__main__":
    main()
