"""Sanity-check the exported ONNX against the Keras model on random input."""
import os
import sys

import numpy as np
import onnxruntime as ort
import tensorflow as tf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from GridTrackNet import GridTrackNet  # noqa: E402

WIDTH, HEIGHT, CHANNELS = 768, 432, 15
ONNX_PATH = os.path.join(REPO_ROOT, "trt", "gridtracknet.onnx")
WEIGHTS = os.path.join(REPO_ROOT, "model_weights.h5")

rng = np.random.default_rng(0)
x = rng.random((2, CHANNELS, HEIGHT, WIDTH), dtype=np.float32)

model = GridTrackNet(5, HEIGHT, WIDTH)
model.load_weights(WEIGHTS)
keras_out = model.predict(x, verbose=0)

sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
onnx_out = sess.run(None, {sess.get_inputs()[0].name: x})[0]

max_abs = float(np.max(np.abs(keras_out - onnx_out)))
mean_abs = float(np.mean(np.abs(keras_out - onnx_out)))
print(f"keras out shape {keras_out.shape}  onnx out shape {onnx_out.shape}")
print(f"max abs diff  = {max_abs:.3e}")
print(f"mean abs diff = {mean_abs:.3e}")
print("MATCH" if max_abs < 1e-4 else "MISMATCH (investigate)")
