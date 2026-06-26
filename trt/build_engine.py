"""Build a TensorRT engine from the GridTrackNet ONNX model.

TensorRT 11 removed the FP16/INT8 BuilderFlags in favour of strongly-typed
networks. To run in half precision we therefore convert the ONNX graph to FP16
(keeping fp32 inputs/outputs for convenience) and build a STRONGLY_TYPED engine,
which honours those FP16 types. A dynamic batch optimization profile lets one
engine serve both the streaming (batch=1) and API (batch=N) paths.

Usage:
    python tensorrt/build_engine.py --fp16 --max-batch 16
    python tensorrt/build_engine.py --no-fp16        # TF32 baseline engine
"""
import argparse
import os

import onnx
import tensorrt as trt
from onnxconverter_common import float16

CHANNELS, HEIGHT, WIDTH = 15, 432, 768


def load_model_bytes(onnx_path, fp16):
    model = onnx.load(onnx_path)
    if fp16:
        # keep_io_types=True -> graph takes/returns fp32, computes internally in fp16.
        model = float16.convert_float_to_float16(model, keep_io_types=True, disable_shape_infer=True)
        print("Converted ONNX graph to FP16 (fp32 I/O preserved)")
    return model.SerializeToString()


def build(onnx_path, engine_path, fp16, min_b, opt_b, max_b, workspace_gb):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    flags = 0
    if fp16:
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)

    parser = trt.OnnxParser(network, logger)
    if not parser.parse(load_model_bytes(onnx_path, fp16)):
        for i in range(parser.num_errors):
            print("ONNX parse error:", parser.get_error(i))
        raise SystemExit(1)

    inp = network.get_input(0)
    print(f"Network input '{inp.name}' shape {inp.shape} dtype {inp.dtype}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    if not fp16:
        config.set_flag(trt.BuilderFlag.TF32)

    profile = builder.create_optimization_profile()
    profile.set_shape(
        inp.name,
        (min_b, CHANNELS, HEIGHT, WIDTH),
        (opt_b, CHANNELS, HEIGHT, WIDTH),
        (max_b, CHANNELS, HEIGHT, WIDTH),
    )
    config.add_optimization_profile(profile)

    print(f"Building {'FP16 strongly-typed' if fp16 else 'TF32'} engine (this can take a minute)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("Engine build failed")

    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"Wrote {engine_path} ({os.path.getsize(engine_path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=os.path.join(here, "gridtracknet.onnx"))
    ap.add_argument("--engine", default=None, help="defaults to gridtracknet_{fp16,tf32}.trt")
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--no-fp16", dest="fp16", action="store_false")
    ap.add_argument("--min-batch", type=int, default=1)
    ap.add_argument("--opt-batch", type=int, default=1)
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--workspace-gb", type=float, default=3.0)
    args = ap.parse_args()

    engine = args.engine or os.path.join(here, f"gridtracknet_{'fp16' if args.fp16 else 'tf32'}.trt")
    build(args.onnx, engine, args.fp16, args.min_batch, args.opt_batch, args.max_batch, args.workspace_gb)
