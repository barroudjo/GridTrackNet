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

`--video_dir` accepts a **single file or a folder** (a folder processes every
`.mp4` that isn't already a `Predicted` output). For each input it writes an
annotated `<name> Predicted TRT_fp16.mp4` plus a per-frame detection CSV
`<name> Predicted TRT_fp16 detections.csv` (`frame,visibility,x,y`; visibility 0
/ `(0,0)` = no ball that frame).

Useful flags:

```bash
--no_video            # skip the annotated video, write only the detections CSV (much faster)
--batch N             # 5-frame units per inference call (default 4; engine supports up to 16)
--decoder cv2|dali    # frame decode+preprocess path (default cv2; see below)
--output_suffix STR   # change the "TRT_fp16" tag in output filenames
--display_trail 0|1   # ball trajectory trail (default 1)
```

`Predict.py` runs a **threaded pipeline** (decode → preprocess → inference →
encode, one thread per stage) so CPU decode/encode overlap GPU inference.
Inference stays on the main thread (the CUDA context is never shared). While
running it prints a live status line — throughput (fps), GPU utilization and
memory (via NVML), and a per-stage time split — and a per-stage bottleneck
breakdown when each video finishes.

`Predict.getPredictions()` (the API) also auto-selects TensorRT on first call.

## Decoders: CPU (cv2) vs GPU (DALI/NVDEC)

`--decoder` selects how frames are decoded and preprocessed:

- **`cv2`** (default) — OpenCV CPU decode + CPU preprocess, in the threaded
  pipeline above.
- **`dali`** — **NVIDIA DALI** hardware-decodes the video (NVDEC), resizes to
  768×432 and converts to BGR float32 CHW **on the GPU**, then feeds the result
  **zero-copy** (device pointer) straight into the TensorRT engine — no host
  round-trip for the input frames. Only the small output grid comes back to the
  CPU. For video output the original full-res frames are also produced and copied
  back for annotation/encode. Requires `--backend trt` (forced automatically) and
  `nvidia-dali-cuda120` (see Notes); implemented in `process_video_dali` using
  `TRTRunner.infer_device`. It runs an overlapped producer (DALI decode+preproc)
  / consumer (TRT inference) pipeline with a small GPU buffer pool, so NVDEC decode
  overlaps inference.

```bash
# Compare the two decoders on the same clip:
.venv/bin/python Predict.py --video_dir "path/to/video.mp4" --decoder cv2  --no_video --batch 4
.venv/bin/python Predict.py --video_dir "path/to/video.mp4" --decoder dali --no_video --batch 4
```

Detections from `dali` match the `cv2` path closely (same model, BGR channel
order preserved), but interpolation/colour-conversion differ slightly from
OpenCV, so coordinates are *near*-identical, not byte-identical (e.g. 257 vs 256
detected frames on a 265-frame clip).

**Measured verdict (RTX 3050 mobile, single 1080p stream): `cv2` wins** (~58 vs
~20 fps). The single mobile NVDEC engine decodes 1080p slower than the
multi-threaded CPU, and even with NVDEC decode overlapping inference, decode +
GPU-preprocess + inference all contend for the one GPU. `cv2`'s structural edge is
that it offloads decode+preprocess to the CPU, leaving the GPU free for inference.
`dali` is expected to pay off on hardware where NVDEC ≥ CPU decode, or with many
concurrent streams. `cv2` stays the default.

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
| `trt_runner.py`    | `TRTRunner`: load engine, run inference (`infer` from host, `infer_device` zero-copy from a GPU pointer); auto-chunks batches > engine max. |
| `benchmark.py`     | Latency/FPS for TRT vs TF-CPU + FP16 accuracy. |
| `validate_video.py`| TRT vs Keras ball coordinates on real video frames. |

The DALI/NVDEC GPU decode path lives in `Predict.py` (`process_video_dali`,
selected with `--decoder dali`), not in a separate script.

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
- **Pipeline bottlenecks (measured):** moving preprocessing onto its own thread
  was the real end-to-end win. For CSV-only runs the bottleneck is inference;
  with video it's the CPU `mp4v` encode (the `--no_video` path skips it). At 1080p
  CPU decode becomes a co-bottleneck — which is what `--decoder dali` (NVDEC)
  addresses.
- **Batching / engine opt profile:** the shipped engine is built with `--opt-batch 1`.
  Sweeping `--batch` and rebuilding at `--opt-batch 4/8` made **no measurable
  difference** on the RTX 3050 — inference time is flat (~3 ms/frame) across
  batch sizes and the opt profile, because throughput is decode/pipeline-bound,
  not inference-bound. `gridtracknet_fp16_opt{4,8}.trt` exist only as experiments;
  the default `gridtracknet_fp16.trt` (opt=1) is the one to use.
- **Laptop power cap (RTX 3050 mobile):** the dGPU can sit at a 30 W enforced
  power limit (default 60 W, max 90 W), which makes it throttle clocks
  (210↔1492 MHz) under load and renders benchmarks meaningless. Check with
  `nvidia-smi -q -d POWER` / `--query-gpu=power.limit`; raise with
  `sudo nvidia-smi -pl 60` (and optionally pin clocks `sudo nvidia-smi -lgc 1500,2100`)
  before timing anything. The dGPU also occasionally wedges in RTD3 suspend
  (`cuInit` → `NO_DEVICE`); only a reboot reliably recovers it.
- **DALI dependency:** `--decoder dali` needs `nvidia-dali-cuda120`
  (`pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda120`).
  It also requires `gast>=0.5` (DALI's autograph import fails on the older 0.4.0
  that TF 2.11 pins; harmless because the TRT path doesn't use TF). DALI's legacy
  video reader needs `skip_vfr_check=True` for some constant-frame-rate files that
  trip its VFR heuristic (already set in `process_video_dali`).
