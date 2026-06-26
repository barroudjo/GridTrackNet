# TensorRT inference for GridTrackNet

Runs the GridTrackNet model through a TensorRT FP16 engine instead of
TensorFlow/Keras. On an RTX 3050 (6 GB, Ampere) this gives **~119 FPS** for the
model forward pass vs **~11 FPS** on the TF-CPU path that this machine is
otherwise stuck on (TF 2.11 can't load the GPU libs it needs) — a **~10.6×**
speedup, with no meaningful accuracy change (100% detection agreement vs Keras
on real frames; mean ball-position difference 0.08 px, well under the model's
4 px tolerance).

## Pipeline

```
model_weights.h5  ──export_onnx.py──▶  gridtracknet.onnx
gridtracknet.onnx ──build_engine.py─▶  gridtracknet_fp16.trt   (FP16, strongly-typed)
Predict.py --backend trt ──▶  uses trt_runner.TRTRunner on the engine
```

Rebuild from scratch (already done once; re-run only if weights change or you
move to different hardware / a different TensorRT version):

```bash
.venv/bin/python trt/export_onnx.py            # h5 -> ONNX (dynamic batch)
.venv/bin/python trt/verify_onnx.py            # ONNX vs Keras numeric check
.venv/bin/python trt/build_engine.py --fp16    # ONNX -> FP16 engine
.venv/bin/python -m trt.benchmark --batch 1    # FPS + accuracy vs TF-CPU
.venv/bin/python -m trt.validate_video         # TRT vs Keras on real frames
```

## Using it

```bash
# Auto: uses TensorRT if trt/gridtracknet_fp16.trt exists, else Keras.
.venv/bin/python Predict.py --video_dir="path/to/video.mp4"

# Force a backend:
.venv/bin/python Predict.py --video_dir="..." --backend trt
.venv/bin/python Predict.py --video_dir="..." --backend keras
```

`Predict.getPredictions()` (the API) also auto-selects TensorRT on first call.

## Environments

Two requirements files, by purpose:

| File | Python | For | TensorFlow |
|------|--------|-----|------------|
| `requirements-trt.txt` | 3.10 or 3.12 | **Inference** — run Predict.py on a built engine | No |
| `requirements.txt`     | 3.10         | **Training / dataset gen / ONNX export** (Train.py, DataGen.py, export_onnx.py) | Yes |

The TensorRT inference path imports no TensorFlow. You only need the training
environment to retrain, regenerate the dataset, or export a NEW `.onnx` from
changed weights. Rebuilding the `.trt` engine from the existing `gridtracknet.onnx`
(e.g. after a TensorRT upgrade or on a new GPU) needs only `onnx` +
`onnxconverter-common` + `tensorrt` — also no TensorFlow.

## Files

| File | Purpose |
|------|---------|
| `export_onnx.py`   | Build the Keras model, load weights, export to ONNX (dynamic batch, fp32 I/O). |
| `verify_onnx.py`   | Confirm ONNX matches Keras numerically. |
| `build_engine.py`  | ONNX → TensorRT engine. FP16 via ONNX-fp16 + STRONGLY_TYPED network. |
| `trt_runner.py`    | `TRTRunner`: load engine, run inference; auto-chunks batches > engine max. |
| `benchmark.py`     | Latency/FPS for TRT vs TF-CPU + FP16 accuracy. |
| `validate_video.py`| TRT vs Keras ball coordinates on real video frames. |

## Notes / gotchas

- **The `.trt` engine is machine-specific** — it's tied to this GPU and the
  installed TensorRT version. Don't commit it or copy it to another box; rebuild
  with `build_engine.py`. (The `.onnx` is portable.)
- **TensorRT 11 removed the `FP16`/`INT8` BuilderFlags.** FP16 is achieved by
  converting the ONNX graph to fp16 and building a `STRONGLY_TYPED` network. See
  `build_engine.py`.
- **Custom weights + TRT:** the engine encodes `model_weights.h5`. To use custom
  weights with TensorRT, re-run `export_onnx.py --weights ... ` then
  `build_engine.py`. `--backend keras --model_dir ...` honors custom weights directly.
- **protobuf:** the ONNX tooling requires `protobuf==3.20.3`, while TF 2.11 pins
  `<3.20`. TF works fine at 3.20.3 in practice (verified), but `pip check` flags
  it. The TRT runtime path itself needs only `tensorrt` + `cuda-python`, not TF.
- **Speedup context:** the big win here is escaping CPU execution. Versus a
  properly configured TF-GPU baseline, expect a smaller (~2–3×) gain. End-to-end
  `Predict.py` throughput is lower than 119 FPS because video decode/resize/encode
  (CPU/OpenCV) are not accelerated.
