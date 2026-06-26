"""Compare TensorRT and Keras backends on real video frames.

Reads frames from a sample video and runs both backends through the exact
Predict.getPredictions path, then reports how often the two agree on the ball
location (and the pixel distance when both detect a ball).
"""
import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import cv2  # noqa: E402
import Predict  # noqa: E402


def read_frames(path, n):
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < n:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    # getPredictions consumes frames in groups of 5
    return frames[: (len(frames) // 5) * 5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=os.path.join(REPO_ROOT, "Sample Video's", "Professional Hardcourt.mp4"))
    ap.add_argument("--frames", type=int, default=100)
    args = ap.parse_args()

    frames = read_frames(args.video, args.frames)
    print(f"Loaded {len(frames)} frames from {os.path.basename(args.video)}")

    Predict.init_backend("trt")
    trt_coords = Predict.getPredictions(frames, isBGRFormat=True)

    Predict.init_backend("keras")
    keras_coords = Predict.getPredictions(frames, isBGRFormat=True)

    both_detect = same_detect = 0
    dists = []
    for (tx, ty), (kx, ky) in zip(trt_coords, keras_coords):
        t_has, k_has = (tx, ty) != (0, 0), (kx, ky) != (0, 0)
        if t_has == k_has:
            same_detect += 1
        if t_has and k_has:
            both_detect += 1
            dists.append(((tx - kx) ** 2 + (ty - ky) ** 2) ** 0.5)

    n = len(trt_coords)
    print(f"\nframes compared            : {n}")
    print(f"detection agreement        : {same_detect}/{n} ({100*same_detect/n:.1f}%)")
    if dists:
        print(f"both detected a ball       : {both_detect} frames")
        print(f"pixel distance (orig res)  : mean {np.mean(dists):.2f}, max {np.max(dists):.2f}")
    print("\nTRT vs Keras coords (first 10):")
    for i in range(min(10, n)):
        print(f"  frame {i:2d}: TRT {trt_coords[i]}   Keras {keras_coords[i]}")


if __name__ == "__main__":
    main()
