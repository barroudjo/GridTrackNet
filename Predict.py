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
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from trt.trt_runner import TRTRunner
    runner = TRTRunner(engine)
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


def _infer_and_decode(units, outputWidth, outputHeight):
    """Run the model on a batch of units (N, 15, HEIGHT, WIDTH) and decode the grid
    output into a flat list of (x, y) ball coordinates, 5 per unit in frame order.
    (0,0) marks 'no ball'. Coordinates are scaled to the output resolution."""
    if _infer is None:
        init_backend()
    t0 = time.time()
    y_pred = _infer(units)
    STAGE_TIMES["infer"] += time.time() - t0

    t0 = time.time()
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
    cv2.destroyAllWindows()
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

    args = parser.parse_args()

    MODEL_DIR = args.model_dir
    DISPLAY_TRAIL = bool(args.display_trail)
    WRITE_VIDEO = not args.no_video
    BATCH_SIZE = max(1, args.batch)

    init_backend(args.backend, weights=MODEL_DIR, engine=args.engine)

    videos = _gather_videos(args.video_dir)
    if not videos:
        print(f"No source videos found at: {args.video_dir}")
        sys.exit(1)

    print(f"Processing {len(videos)} video(s):")
    for v in videos:
        print(f"  - {os.path.basename(v)}")

    for v in videos:
        process_video(v, display_trail=DISPLAY_TRAIL, output_suffix=args.output_suffix,
                      write_video=WRITE_VIDEO, batch_size=BATCH_SIZE)

    print("\nAll videos processed.")