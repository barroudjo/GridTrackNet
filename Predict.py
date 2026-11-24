import cv2
import argparse
import os
import sys
import time
import queue
import threading
import numpy as np

WIDTH = 768
HEIGHT = 432
IMGS_PER_INSTANCE = 5
GRID_COLS = 48
GRID_ROWS = 27
GRID_SIZE_COL = WIDTH/GRID_COLS
GRID_SIZE_ROW = HEIGHT/GRID_ROWS

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(os.getcwd(),"model_weights.h5")
DEFAULT_ENGINE = os.path.join(REPO_ROOT, "trt", "gridtracknet_fp16.trt")

# Inference backend: a callable mapping a float32 (N,15,432,768) batch to the
# (N,15,27,48) grid output. Chosen lazily so the TensorRT path needs no TensorFlow.
_infer = None
_backend_name = None
# The underlying TRTRunner when the TensorRT backend is active, exposed so the
# DALI decoder can feed GPU buffers zero-copy via runner.infer_device().
_trt_runner = None

# Cumulative per-stage wall-clock (seconds), used to locate pipeline bottlenecks.
# Reset per video by process_video(); updated by getPredictions() and the main loop.
STAGE_TIMES = {"decode": 0.0, "preprocess": 0.0, "infer": 0.0, "postprocess": 0.0, "write": 0.0}
STAGE_ORDER = ["decode", "preprocess", "infer", "postprocess", "write"]


def _reset_stage_times():
    for k in STAGE_TIMES:
        STAGE_TIMES[k] = 0.0


def _init_keras_backend(weights):
    import tensorflow as tf  # noqa: F401
    from GridTrackNet import GridTrackNet
    m = GridTrackNet(IMGS_PER_INSTANCE, HEIGHT, WIDTH)
    m.load_weights(weights)
    return lambda units: m.predict(units, batch_size=len(units), verbose=0)


def _init_trt_backend(engine):
    global _trt_runner
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from trt.trt_runner import TRTRunner
    runner = TRTRunner(engine)
    _trt_runner = runner
    return lambda units: runner.infer(units)


def init_backend(backend="auto", weights=None, engine=DEFAULT_ENGINE):
    """Select the inference backend: 'auto' | 'trt' | 'keras'.

    'auto' uses TensorRT when an engine file is present, otherwise Keras.
    Explicit 'trt' raises if the engine can't be loaded instead of falling back.
    """
    global _infer, _backend_name
    weights = weights or MODEL_DIR

    if backend in ("auto", "trt") and (backend == "trt" or os.path.exists(engine)):
        try:
            _infer = _init_trt_backend(engine)
            _backend_name = "TensorRT"
            print(f"[Predict] backend = TensorRT ({engine})")
            return
        except Exception as e:
            if backend == "trt":
                raise
            print(f"[Predict] TensorRT unavailable ({e}); falling back to Keras")

    try:
        _infer = _init_keras_backend(weights)
    except ImportError as e:
        raise ImportError(
            "Keras backend needs TensorFlow, which isn't installed in this "
            "inference-only environment (requirements-trt.txt). Use --backend trt "
            "with a built engine (trt/gridtracknet_fp16.trt), or install the "
            "training environment (requirements.txt)."
        ) from e
    _backend_name = "Keras"
    print("[Predict] backend = TensorFlow/Keras")


def _preprocess_unit(frames, isBGRFormat=False):
    """Turn 5 BGR frames into one model unit: float32 (15, HEIGHT, WIDTH) in [0,1],
    channels-first with the 5 frames' RGB planes concatenated. Pure CPU work, so it
    can run off the inference thread."""
    unit = []
    for frame in frames:
        if(isBGRFormat):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.moveaxis(frame, -1, 0)
        unit.append(frame[0])
        unit.append(frame[1])
        unit.append(frame[2])
    unit = np.asarray(unit, dtype=np.float32)
    unit /= 255
    return unit


def decodeGrid(y_pred, outputWidth, outputHeight):
    """Decode the raw model output (N, 15, GRID_ROWS, GRID_COLS) into a flat list of
    (x, y) ball coordinates, 5 per unit in frame order. (0,0) marks 'no ball'.
    Pure NumPy on the host output grid, so it's shared by the CPU and DALI paths."""
    y_pred = np.split(y_pred, IMGS_PER_INSTANCE, axis=1)
    y_pred = np.stack(y_pred, axis=2)
    y_pred = np.moveaxis(y_pred, 1, -1)

    confGrid, xOffsetGrid, yOffsetGrid = np.split(y_pred, 3, axis=-1)

    confGrid = np.squeeze(confGrid, axis=-1)
    xOffsetGrid = np.squeeze(xOffsetGrid, axis=-1)
    yOffsetGrid = np.squeeze(yOffsetGrid, axis=-1)

    ballCoordinates = []
    for i in range(0, confGrid.shape[0]):
        for j in range(0, confGrid.shape[1]):
            currConfGrid = confGrid[i][j]
            currXOffsetGrid = xOffsetGrid[i][j]
            currYOffsetGrid = yOffsetGrid[i][j]

            maxConfVal = np.max(currConfGrid)
            predRow, predCol = np.unravel_index(np.argmax(currConfGrid), currConfGrid.shape)

            threshold = 0.5
            predHasBall = maxConfVal >= threshold

            xOffset = currXOffsetGrid[predRow][predCol]
            yOffset = currYOffsetGrid[predRow][predCol]

            xPred = int((xOffset + predCol) * GRID_SIZE_COL)
            yPred = int((yOffset + predRow) * GRID_SIZE_ROW)

            if(predHasBall):
                ballCoordinates.append((int((xPred/WIDTH)*outputWidth), int((yPred/HEIGHT)*outputHeight)))
            else:
                ballCoordinates.append((0,0))

    return ballCoordinates


def _infer_and_decode(units, outputWidth, outputHeight):
    """Run the model on a batch of units (N, 15, HEIGHT, WIDTH) and decode the grid
    output into a flat list of (x, y) ball coordinates, 5 per unit in frame order."""
    if _infer is None:
        init_backend()
    t0 = time.time()
    y_pred = _infer(units)
    STAGE_TIMES["infer"] += time.time() - t0

    t0 = time.time()
    ballCoordinates = decodeGrid(y_pred, outputWidth, outputHeight)
    STAGE_TIMES["postprocess"] += time.time() - t0
    return ballCoordinates


def getPredictions(frames, isBGRFormat = False):
    outputHeight = frames[0].shape[0]
    outputWidth = frames[0].shape[1]

    t0 = time.time()
    units = []
    for i in range(0, len(frames), 5):
        batch = frames[i:i+5]
        if len(batch) == 5:
            units.append(_preprocess_unit(batch, isBGRFormat))
    units = np.stack(units, axis=0)
    STAGE_TIMES["preprocess"] += time.time() - t0

    return _infer_and_decode(units, outputWidth, outputHeight)


OUTPUT_SUFFIX = "TRT_fp16"


class _GPUStats:
    """Lightweight NVML poller for live GPU utilization / memory. No-ops if NVML
    isn't available (e.g. CPU/Keras runs), so callers never need to guard it."""

    def __init__(self):
        self._nvml = None
        self._handle = None
        try:
            from cuda.bindings import nvml
            nvml.init_v2()
            self._handle = nvml.device_get_handle_by_index_v2(0)
            self._nvml = nvml
        except Exception:
            self._nvml = None

    def sample(self):
        """Return (gpu_util_pct, mem_used_mb, mem_total_mb) or None if unavailable."""
        if self._nvml is None:
            return None
        try:
            u = self._nvml.device_get_utilization_rates(self._handle)
            m = self._nvml.device_get_memory_info_v2(self._handle)
            return u.gpu, m.used // (1024 * 1024), m.total // (1024 * 1024)
        except Exception:
            return None

    def close(self):
        if self._nvml is not None:
            try:
                self._nvml.shutdown()
            except Exception:
                pass
            self._nvml = None


def process_video(video_path, display_trail=True, output_suffix=OUTPUT_SUFFIX, write_video=True,
                  batch_size=4):
    """Run inference on a single video, always writing a per-frame detection CSV
    and, when write_video is True, an annotated video alongside it. Skipping the
    video removes the dominant encode cost. Returns (output_path, detections_path);
    output_path is None when no video is written.

    batch_size is the number of 5-frame units fed to the model per inference call;
    larger values amortize per-call overhead and raise GPU utilization."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if(fps >= 57 and fps <= 62):
        numFramesSkip = 2
    elif (fps >= 22 and fps <= 32):
        numFramesSkip = 1
    else:
        print(f"ERROR: '{video_path}' is not 30FPS or 60FPS (got {fps}); skipping")
        cap.release()
        return None

    totalFrames = (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // numFramesSkip)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    directory, filename = os.path.split(video_path)
    name, extension = os.path.splitext(filename)
    detections_path = os.path.join(directory, f"{name} Predicted {output_suffix} detections.csv")

    output_path = None
    video_writer = None
    if write_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_filename = f"{name} Predicted {output_suffix}{extension}"
        output_path = os.path.join(directory, output_filename)
        if(numFramesSkip == 1):
            outputFPS = fps
        else:
            outputFPS = 30
        video_writer = cv2.VideoWriter(output_path, fourcc, outputFPS, (frame_width, frame_height))

    # One line per output (predicted) frame: frame index, visibility flag, x, y.
    # Coordinates are in the source video's pixel space; (0,0)/visibility 0 means
    # no ball was detected in that frame.
    det_file = open(detections_path, "w")
    det_file.write("frame,visibility,x,y\n")
    outFrameIdx = 0

    dest = output_filename if write_video else os.path.basename(detections_path) + " (CSV only)"
    print(f"\n[{filename}] -> {dest}")

    _reset_stage_times()
    gpu = _GPUStats()

    # Four-stage pipeline, one thread per CPU stage so they overlap the GPU:
    #   decoder   -> raw_q    : read frames, group into 5-frame units
    #   preprocessor -> pre_q : resize/convert/normalize each unit (CPU-heavy)
    #   main loop           : batch units, run ONE inference call, decode, write CSV
    #   writer              : annotate + encode the output video
    # Inference stays on the main thread so the CUDA context is never shared.
    # Each STAGE_TIMES key is written by exactly one thread, so the += is race-free.
    SENTINEL = None
    raw_q = queue.Queue(maxsize=8)
    pre_q = queue.Queue(maxsize=batch_size + 2)
    write_q = queue.Queue(maxsize=8) if write_video else None

    def decoder():
        idx = 0
        group = []
        while True:
            t = time.time()
            ret, frame = cap.read()
            STAGE_TIMES["decode"] += time.time() - t
            if not ret:
                break
            if idx % numFramesSkip == 0:
                group.append(frame)
                if len(group) == 5:
                    raw_q.put(group)
                    group = []
            idx += 1
        raw_q.put(SENTINEL)

    def preprocessor():
        while True:
            group = raw_q.get()
            if group is SENTINEL:
                break
            t = time.time()
            unit = _preprocess_unit(group, True)
            STAGE_TIMES["preprocess"] += time.time() - t
            pre_q.put((group, unit))
        pre_q.put(SENTINEL)

    def writer():
        history = []
        while True:
            item = write_q.get()
            if item is SENTINEL:
                break
            unit_frames, coords = item
            t = time.time()
            history.extend(coords)
            for i, frame in enumerate(unit_frames):
                if i < len(coords):
                    if display_trail:
                        if len(history) >= 15:
                            for j in range(7, -1, -2):
                                k = len(history) - 5 - j + i
                                cv2.circle(frame, history[k], 4, (0, 255, 255), -1)
                    else:
                        cv2.circle(frame, coords[i], 8, (0, 0, 255), 4)
                video_writer.write(frame)
            STAGE_TIMES["write"] += time.time() - t

    def flush_batch(batch_groups, batch_units):
        """Run one inference over the accumulated units, then emit CSV + video work."""
        nonlocal outFrameIdx
        if not batch_units:
            return
        units = np.stack(batch_units, axis=0)
        coords = _infer_and_decode(units, frame_width, frame_height)  # 5 per unit, in order
        for g, group in enumerate(batch_groups):
            group_coords = coords[g * 5:(g + 1) * 5]
            for x, y in group_coords:
                visibility = 0 if (x == 0 and y == 0) else 1
                det_file.write(f"{outFrameIdx},{visibility},{x},{y}\n")
                outFrameIdx += 1
            if write_video:
                write_q.put((group, group_coords))
        batch_groups.clear()
        batch_units.clear()

    start_time = time.time()
    last_report = start_time
    last_frames = 0
    status = ""  # cached suffix (fps + GPU + stage split), refreshed on a throttle

    dec_thread = threading.Thread(target=decoder, daemon=True)
    pre_thread = threading.Thread(target=preprocessor, daemon=True)
    dec_thread.start()
    pre_thread.start()
    wr_thread = None
    if write_video:
        wr_thread = threading.Thread(target=writer, daemon=True)
        wr_thread.start()

    batch_groups = []
    batch_units = []
    while True:
        item = pre_q.get()
        if item is SENTINEL:
            flush_batch(batch_groups, batch_units)  # tail < batch_size
            print("\nDone")
            break

        group, unit = item
        batch_groups.append(group)
        batch_units.append(unit)
        if len(batch_units) >= batch_size:
            flush_batch(batch_groups, batch_units)

        # Refresh the throughput / GPU stats about twice a second to keep the
        # carriage-return status line cheap and readable. With the pipeline the
        # stage split shows each thread's share of busy time, so it sums to >100%
        # (overlap) — the largest stage is the throughput-limiting one.
        now = time.time()
        if now - last_report >= 0.5:
            fps_now = (outFrameIdx - last_frames) / (now - last_report)
            last_report, last_frames = now, outFrameIdx
            sample = gpu.sample()
            total_stage = sum(STAGE_TIMES.values()) or 1.0
            split = " ".join(
                f"{k[:3]} {int(STAGE_TIMES[k] / total_stage * 100):2d}%" for k in STAGE_ORDER
            )
            if sample is not None:
                util, used, total = sample
                status = f" | {fps_now:6.1f} fps | GPU {util:3d}% | Mem {used}/{total} MB | {split}"
            else:
                status = f" | {fps_now:6.1f} fps | {split}"

        percentage = outFrameIdx*100/totalFrames if totalFrames else 0
        print('Exporting...[%3d%%]%s   \r' % (int(percentage), status), end="")

    # Drain the pipeline: decoder + preprocessor already hit EOF; flush the writer.
    dec_thread.join()
    pre_thread.join()
    if write_video:
        write_q.put(SENTINEL)
        wr_thread.join()

    elapsed = time.time() - start_time
    avg_fps = outFrameIdx / elapsed if elapsed > 0 else 0
    print(f"Processed {outFrameIdx} frames in {elapsed:.1f}s ({avg_fps:.1f} fps avg, batch={batch_size})")

    # Threaded breakdown: each stage runs on its own thread, so percentages are of
    # wall-clock and overlap (they can exceed 100%). The stage with the highest
    # share is the pipeline bottleneck; wall-clock ~= that stage + fill/drain.
    n = max(outFrameIdx, 1)
    threads = [
        ("decode", "decode     (decoder thread)     ", STAGE_TIMES["decode"]),
        ("preprocess", "preprocess (preprocessor thread)", STAGE_TIMES["preprocess"]),
        ("inference", "inference  (main thread)        ", STAGE_TIMES["infer"] + STAGE_TIMES["postprocess"]),
        ("write", "write      (writer thread)      ", STAGE_TIMES["write"]),
    ]
    bottleneck = max(threads, key=lambda t: t[2])[0]
    print(f"  Pipeline breakdown (bottleneck: {bottleneck}):")
    for _, label, secs in threads:
        pct = secs / elapsed * 100 if elapsed > 0 else 0
        print(f"    {label} {secs:7.2f}s  {pct:5.1f}% busy  {secs / n * 1000:6.2f} ms/frame")
    print("    sub-stages:  " + "  ".join(
        f"{k}={STAGE_TIMES[k] / n * 1000:.2f}ms" for k in STAGE_ORDER))

    gpu.close()
    cap.release()
    if video_writer is not None:
        video_writer.release()
    det_file.close()
    #cv2.destroyAllWindows()
    return output_path, detections_path


def process_video_dali(video_path, display_trail=True, output_suffix=OUTPUT_SUFFIX,
                       write_video=True, batch_size=4):
    """GPU-resident variant of process_video: NVIDIA DALI hardware-decodes (NVDEC)
    and preprocesses on the GPU, then the units are fed zero-copy into TensorRT.
    For video output the original full-res frames are also produced (copied back to
    the host) so the same annotation + encode can run. Requires the TensorRT backend
    (an engine) and nvidia-dali; otherwise identical outputs to process_video."""
    if _trt_runner is None:
        raise RuntimeError(
            "--decoder dali requires the TensorRT backend. Build an engine and run "
            "with --backend trt (the DALI path feeds GPU buffers via infer_device)."
        )
    from nvidia.dali import pipeline_def, fn, types
    from cuda.bindings import runtime as cudart

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if 57 <= fps <= 62:
        numFramesSkip = 2
    elif 22 <= fps <= 32:
        numFramesSkip = 1
    else:
        print(f"ERROR: '{video_path}' is not 30FPS or 60FPS (got {fps}); skipping")
        return None

    batch_size = max(1, min(batch_size, _trt_runner.max_batch))

    directory, filename = os.path.split(video_path)
    name, extension = os.path.splitext(filename)
    detections_path = os.path.join(directory, f"{name} Predicted {output_suffix} detections.csv")

    output_path = None
    video_writer = None
    if write_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_filename = f"{name} Predicted {output_suffix}{extension}"
        output_path = os.path.join(directory, output_filename)
        outputFPS = int(fps) if numFramesSkip == 1 else 30
        video_writer = cv2.VideoWriter(output_path, fourcc, outputFPS, (frame_width, frame_height))

    det_file = open(detections_path, "w")
    det_file.write("frame,visibility,x,y\n")
    dest = output_filename if write_video else os.path.basename(detections_path) + " (CSV only)"
    print(f"\n[{filename}] -> {dest}  [decoder=dali/NVDEC]")

    need_frames = write_video

    # NVDEC-decode 5-frame units (stride = frame-skip, step = 5*stride for
    # non-overlapping units), resize to the model size, convert RGB->BGR to match
    # the CPU path, normalize to CHW float. For video we also emit the original
    # full-res frames (as BGR) for annotation/encode.
    @pipeline_def(batch_size=batch_size, num_threads=3, device_id=0, prefetch_queue_depth=2)
    def _pipe():
        full = fn.readers.video(
            device="gpu", filenames=[video_path], sequence_length=5,
            stride=numFramesSkip, step=5 * numFramesSkip, image_type=types.RGB,
            skip_vfr_check=True, pad_last_batch=True, name="reader",
        )
        small = fn.resize(full, resize_x=WIDTH, resize_y=HEIGHT)
        small = fn.color_space_conversion(small, image_type=types.RGB, output_type=types.BGR)
        units = fn.crop_mirror_normalize(
            small, dtype=types.FLOAT, output_layout="FCHW", mean=[0.0], std=[255.0])
        units = fn.reshape(units, shape=[15, HEIGHT, WIDTH])
        if need_frames:
            return units, fn.color_space_conversion(full, image_type=types.RGB, output_type=types.BGR)
        return units

    pipe = _pipe()
    pipe.build()
    seq_count = int(pipe.epoch_size("reader"))   # number of 5-frame units in one pass
    iters = (seq_count + batch_size - 1) // batch_size
    totalFrames = seq_count * 5

    # Writer thread (annotate + encode) so the CPU encode overlaps GPU inference.
    SENTINEL = None
    write_q = queue.Queue(maxsize=8) if write_video else None
    t_write = {"s": 0.0}

    def writer():
        history = []
        while True:
            item = write_q.get()
            if item is SENTINEL:
                break
            unit_frames, coords = item
            t = time.time()
            history.extend(coords)
            for i, frame in enumerate(unit_frames):
                if i < len(coords):
                    if display_trail:
                        if len(history) >= 15:
                            for j in range(7, -1, -2):
                                k = len(history) - 5 - j + i
                                cv2.circle(frame, history[k], 4, (0, 255, 255), -1)
                    else:
                        cv2.circle(frame, coords[i], 8, (0, 0, 255), 4)
                video_writer.write(frame)
            t_write["s"] += time.time() - t

    wr_thread = None
    if write_video:
        wr_thread = threading.Thread(target=writer, daemon=True)
        wr_thread.start()

    gpu = _GPUStats()

    def _cuda(ret):
        err, *rest = ret if isinstance(ret, (tuple, list)) else (ret,)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA error: {int(err)}")
        return rest[0] if len(rest) == 1 else None

    # GPU buffer pool: the producer copies each batch of units into a free pool
    # buffer (cheap device-to-device), so DALI can immediately reuse its own output
    # buffer to decode the NEXT batch while TensorRT still reads this one. This is
    # what lets NVDEC decode overlap TRT inference instead of running serially.
    unit_nbytes = batch_size * 15 * HEIGHT * WIDTH * 4
    POOL = 4
    pool = [_cuda(cudart.cudaMalloc(unit_nbytes)) for _ in range(POOL)]
    free_q = queue.Queue()
    for s in range(POOL):
        free_q.put(s)
    work_q = queue.Queue(maxsize=POOL)
    DtoD = cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice

    t_dali = {"s": 0.0}      # producer-thread busy time (decode + preproc + copy)
    t_infer = 0.0            # consumer (main) inference time
    t_decode = 0.0           # grid decode

    def producer():
        produced = 0
        for _ in range(iters):
            t = time.time()
            out = pipe.run()
            units_t = out[0].as_tensor()
            cudart.cudaDeviceSynchronize()   # DALI ready; inference still overlaps (separate thread)
            slot = free_q.get()              # blocks (backpressure) until a buffer frees
            _cuda(cudart.cudaMemcpy(pool[slot], units_t.data_ptr(), unit_nbytes, DtoD))
            shape = tuple(int(s) for s in units_t.shape())
            # Copy the full-res frames to host once (decoupled from DALI's reused
            # buffer); the writer slices them later.
            frames = np.array(out[1].as_cpu().as_array()) if write_video else None
            t_dali["s"] += time.time() - t
            real = min(batch_size, seq_count - produced)   # drop padded tail
            produced += real
            work_q.put((slot, shape, frames, real))
        work_q.put(SENTINEL)

    prod_thread = threading.Thread(target=producer, daemon=True)
    prod_thread.start()

    outFrameIdx = 0
    last_report = time.time()
    last_frames = 0
    status = ""
    start_time = time.time()
    while True:
        item = work_q.get()
        if item is SENTINEL:
            break
        slot, shape, frames, real = item

        t = time.time()
        y_pred = _trt_runner.infer_device(pool[slot], shape)
        t_infer += time.time() - t
        free_q.put(slot)                     # input consumed; buffer reusable now

        t = time.time()
        coords = decodeGrid(y_pred, frame_width, frame_height)
        t_decode += time.time() - t

        if write_video:
            for g in range(real):
                group = [frames[g, i] for i in range(5)]   # frames is a fresh per-batch copy
                write_q.put((group, coords[g * 5:(g + 1) * 5]))
        for x, yy in coords[:real * 5]:
            det_file.write(f"{outFrameIdx},{0 if (x == 0 and yy == 0) else 1},{x},{yy}\n")
            outFrameIdx += 1

        now = time.time()
        if now - last_report >= 0.5:
            fps_now = (outFrameIdx - last_frames) / (now - last_report)
            last_report, last_frames = now, outFrameIdx
            sample = gpu.sample()
            if sample is not None:
                util, used, total = sample
                status = f" | {fps_now:6.1f} fps | GPU {util:3d}% | Mem {used}/{total} MB"
            else:
                status = f" | {fps_now:6.1f} fps"
        pct = outFrameIdx * 100 / totalFrames if totalFrames else 0
        print('Exporting...[%3d%%]%s   \r' % (int(pct), status), end="")

    prod_thread.join()
    if write_video:
        write_q.put(SENTINEL)
        wr_thread.join()
    print("\nDone")

    for b in pool:
        cudart.cudaFree(b)

    elapsed = time.time() - start_time
    n = max(outFrameIdx, 1)
    print(f"Processed {outFrameIdx} frames in {elapsed:.1f}s "
          f"({outFrameIdx/elapsed:.1f} fps avg, batch={batch_size}, decoder=dali, overlapped)")
    # Stages run on separate threads, so busy% overlaps (can exceed 100%); the
    # largest is the throughput-limiting one (NVDEC decode for this workload).
    print(f"  dali decode+preproc (producer) {t_dali['s']:6.2f}s  {t_dali['s']/elapsed*100:5.1f}% busy  {t_dali['s']/n*1000:6.2f} ms/fr")
    print(f"  inference           (consumer) {t_infer:6.2f}s  {t_infer/elapsed*100:5.1f}% busy  {t_infer/n*1000:6.2f} ms/fr")
    if write_video:
        print(f"  write               (writer)   {t_write['s']:6.2f}s  {t_write['s']/elapsed*100:5.1f}% busy  {t_write['s']/n*1000:6.2f} ms/fr")

    gpu.close()
    det_file.close()
    if video_writer is not None:
        video_writer.release()
    #cv2.destroyAllWindows()
    return output_path, detections_path


def _gather_videos(path):
    """Resolve a path to a list of source .mp4 videos. A directory yields every
    .mp4 that isn't an already-predicted output."""
    if os.path.isdir(path):
        vids = []
        for f in sorted(os.listdir(path)):
            if not f.lower().endswith(".mp4"):
                continue
            if "Predicted" in f:  # skip previously generated outputs
                continue
            vids.append(os.path.join(path, f))
        return vids
    return [path]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument Parser for GridTrackNet')

    parser.add_argument('--video_dir', required=True, type=str, help="Path to a .mp4 video file or a folder of videos.")
    parser.add_argument('--model_dir', required=False, default=os.path.join(os.getcwd(),"model_weights.h5"), type=str, help="Path to saved Tensorflow model.")
    parser.add_argument('--display_trail', required=False, default=1, type=int, help="Output a visible trail of the ball's trajectory. Default = 1")
    parser.add_argument('--backend', required=False, default="auto", choices=["auto", "trt", "keras"], help="Inference backend. 'auto' uses TensorRT if an engine exists, else Keras. Default = auto")
    parser.add_argument('--engine', required=False, default=DEFAULT_ENGINE, type=str, help="Path to the TensorRT engine (.trt) for the 'trt'/'auto' backends.")
    parser.add_argument('--output_suffix', required=False, default=OUTPUT_SUFFIX, type=str, help=f"Suffix appended to output file names. Default = {OUTPUT_SUFFIX}")
    parser.add_argument('--no_video', required=False, action="store_true", help="Skip writing the annotated video; only produce the detections .csv (much faster).")
    parser.add_argument('--batch', required=False, default=4, type=int, help="Number of 5-frame units per inference call (engine supports up to 16). Default = 4. NOTE: the current engine is built with opt batch=1, so larger batches give little/no gain until decode moves off-CPU or the engine is rebuilt with a larger opt profile.")
    parser.add_argument('--decoder', required=False, default="cv2", choices=["cv2", "dali"], help="Frame decode + preprocess path. 'cv2' = CPU OpenCV pipeline (default). 'dali' = GPU NVDEC decode + GPU preprocess fed zero-copy into TensorRT (requires --backend trt and nvidia-dali).")

    args = parser.parse_args()

    MODEL_DIR = args.model_dir
    DISPLAY_TRAIL = bool(args.display_trail)
    WRITE_VIDEO = not args.no_video
    BATCH_SIZE = max(1, args.batch)

    # The DALI decoder feeds GPU buffers straight into the engine, so it requires
    # the TensorRT backend regardless of --backend.
    backend = "trt" if args.decoder == "dali" else args.backend
    init_backend(backend, weights=MODEL_DIR, engine=args.engine)

    videos = _gather_videos(args.video_dir)
    if not videos:
        print(f"No source videos found at: {args.video_dir}")
        sys.exit(1)

    print(f"Processing {len(videos)} video(s) [decoder={args.decoder}]:")
    for v in videos:
        print(f"  - {os.path.basename(v)}")

    run = process_video_dali if args.decoder == "dali" else process_video
    for v in videos:
        run(v, display_trail=DISPLAY_TRAIL, output_suffix=args.output_suffix,
            write_video=WRITE_VIDEO, batch_size=BATCH_SIZE)

    print("\nAll videos processed.")
