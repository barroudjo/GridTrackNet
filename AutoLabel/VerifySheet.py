"""Contact sheets for eyeballing auto-generated labels.

Two sheets are produced:
  <out>_hits.jpg    crops centred on frames labelled VISIBLE - the cross should
                    sit on the ball.
  <out>_misses.jpg  crops centred on the interpolated position of frames
                    labelled INVISIBLE - these should show no ball (otherwise
                    the labels teach the model false negatives).
"""
import argparse
import csv
import os

import cv2
import numpy as np

WIDTH, HEIGHT = 1280, 720


def read_labels(path):
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[int(row["Frame"])] = (int(row["Visibility"]), float(row["X"]), float(row["Y"]))
    return out


def interpolate(labels, t):
    prev = next((k for k in range(t - 1, t - 60, -1) if labels.get(k, (0,))[0] == 1), None)
    nxt = next((k for k in range(t + 1, t + 60) if labels.get(k, (0,))[0] == 1), None)
    if prev is None and nxt is None:
        return (WIDTH / 2, HEIGHT / 2)
    if prev is None:
        return labels[nxt][1:]
    if nxt is None:
        return labels[prev][1:]
    a = (t - prev) / float(nxt - prev)
    return (labels[prev][1] + a * (labels[nxt][1] - labels[prev][1]),
            labels[prev][2] + a * (labels[nxt][2] - labels[prev][2]))


def make_sheet(video, frames_and_points, out_path, size=110, cols=12, colour=(0, 0, 255)):
    if not frames_and_points:
        return
    cap = cv2.VideoCapture(video)
    tiles = []
    for t, (x, y) in frames_and_points:
        cap.set(cv2.CAP_PROP_POS_FRAMES, t)
        ok, f = cap.read()
        if not ok:
            continue
        f = cv2.resize(f, (WIDTH, HEIGHT))
        h = size // 2
        pad = cv2.copyMakeBorder(f, h, h, h, h, cv2.BORDER_CONSTANT)
        tile = pad[int(y):int(y) + size, int(x):int(x) + size].copy()
        tile = cv2.resize(tile, (size * 2, size * 2), interpolation=cv2.INTER_NEAREST)
        cv2.drawMarker(tile, (size, size), colour, cv2.MARKER_CROSS, 24, 1)
        cv2.putText(tile, str(t), (3, 18), 0, 0.55, (0, 255, 255), 1)
        tiles.append(tile)
    cap.release()
    blank = np.zeros_like(tiles[0])
    rows = [np.hstack(tiles[i:i + cols] + [blank] * (cols - len(tiles[i:i + cols])))
            for i in range(0, len(tiles), cols)]
    cv2.imwrite(out_path, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("wrote %s (%d tiles)" % (out_path, len(tiles)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=10 ** 9)
    ap.add_argument("--max_tiles", type=int, default=48)
    ap.add_argument("--frame_offset", type=int, default=0,
                    help="added to label indices to get video frame numbers")
    args = ap.parse_args()

    labels = read_labels(args.labels)
    rng = [t for t in sorted(labels) if args.start <= t <= args.end]
    hits = [t for t in rng if labels[t][0] == 1]
    misses = [t for t in rng if labels[t][0] == 0]

    def sample(seq):
        if len(seq) <= args.max_tiles:
            return seq
        step = len(seq) / float(args.max_tiles)
        return [seq[int(i * step)] for i in range(args.max_tiles)]

    make_sheet(args.video, [(t + args.frame_offset, labels[t][1:]) for t in sample(hits)],
               args.out_prefix + "_hits.jpg")
    make_sheet(args.video, [(t + args.frame_offset, interpolate(labels, t)) for t in sample(misses)],
               args.out_prefix + "_misses.jpg", colour=(255, 0, 255))
    print("visible %d / %d frames (%.1f%%)" % (len(hits), len(rng), 100.0 * len(hits) / max(1, len(rng))))


if __name__ == "__main__":
    main()
