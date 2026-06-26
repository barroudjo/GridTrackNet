"""Export the GridTrackNet Keras model to ONNX.

Builds the model from GridTrackNet.py, loads model_weights.h5, and writes an
ONNX graph with a dynamic batch dimension (input: N x 15 x 432 x 768).

Usage:
    python trt/export_onnx.py \
        --weights model_weights.h5 \
        --out trt/gridtracknet.onnx \
        --opset 17
"""
import argparse
import os
import sys

import numpy as np
import tensorflow as tf
import tf2onnx
import onnx

# Allow importing GridTrackNet.py from the repo root regardless of CWD.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from GridTrackNet import GridTrackNet  # noqa: E402

WIDTH = 768
HEIGHT = 432
IMGS_PER_INSTANCE = 5
CHANNELS = IMGS_PER_INSTANCE * 3  # 15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(REPO_ROOT, "model_weights.h5"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "trt", "gridtracknet.onnx"))
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    model = GridTrackNet(IMGS_PER_INSTANCE, HEIGHT, WIDTH)
    model.load_weights(args.weights)
    print(f"Loaded weights from {args.weights}")
    print(f"Model input shape:  {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")

    # Dynamic batch dimension so one engine serves both the streaming (batch=1)
    # and the API (batch=N) call paths in Predict.py.
    spec = (tf.TensorSpec((None, CHANNELS, HEIGHT, WIDTH), tf.float32, name="input"),)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model_proto, _ = tf2onnx.convert.from_keras(
        model, input_signature=spec, opset=args.opset, output_path=args.out
    )

    inp = model_proto.graph.input[0]
    out = model_proto.graph.output[0]
    print(f"\nWrote {args.out}")
    print(f"ONNX input:  {inp.name}  {[d.dim_param or d.dim_value for d in inp.type.tensor_type.shape.dim]}")
    print(f"ONNX output: {out.name}  {[d.dim_param or d.dim_value for d in out.type.tensor_type.shape.dim]}")

    onnx.checker.check_model(args.out)
    print("onnx.checker: OK")


if __name__ == "__main__":
    main()
