"""Draw ball labels onto extracted frames.

Reads either a Labels.csv (Frame,Visibility,X,Y) or a LabelStore JSON and writes
annotated copies of the frames: a circle at the ball, a short trail of recent
positions, and a banner stating whether the frame is labelled VISIBLE,
INVISIBLE (looked at, no ball in play) or UNLABELLED.
"""
import argparse
import csv
import json
import os

import cv2

TRAIL = 12


def load_labels(path):
    """-> {frame: (x, y) or None}; frames absent from the file are unlabelled."""
    out = {}
    if path.endswith(".json"):
        for k, v in json.load(open(path)).items():
            out[int(k)] = tuple(v) if v else None
    else:
        with open(path) as f:
            for row in csv.DictReader(f):
                vis = int(row["Visibility"])
                out[int(row["Frame"])] = (float(row["X"]), float(row["Y"])) if vis else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=10 ** 9)
    ap.add_argument("--only_labelled", action="store_true")
    args = ap.parse_args()

    labels = load_labels(args.labels)
    os.makedirs(args.out_dir, exist_ok=True)
    frames = sorted(int(f[:-4]) for f in os.listdir(args.frames_dir) if f.endswith(".png"))
    frames = [f for f in frames if args.start <= f <= args.end]
    if args.only_labelled:
        frames = [f for f in frames if f in labels]

    written = 0
    for t in frames:
        img = cv2.imread(os.path.join(args.frames_dir, "%d.png" % t))
        if img is None:
            continue
        trail = [labels.get(k) for k in range(t - TRAIL, t) if labels.get(k)]
        for i, p in enumerate(trail):
            r = 1 + int(3 * (i + 1) / len(trail))
            cv2.circle(img, (int(p[0]), int(p[1])), r, (0, 200, 255), -1, cv2.LINE_AA)
        if t not in labels:
            state, colour = "UNLABELLED", (160, 160, 160)
        elif labels[t] is None:
            state, colour = "no ball in play", (200, 120, 255)
        else:
            state, colour = "ball %d,%d" % labels[t], (0, 0, 255)
            x, y = int(labels[t][0]), int(labels[t][1])
            cv2.circle(img, (x, y), 16, colour, 2, cv2.LINE_AA)
            cv2.line(img, (x - 26, y), (x - 8, y), colour, 1, cv2.LINE_AA)
            cv2.line(img, (x + 8, y), (x + 26, y), colour, 1, cv2.LINE_AA)
            cv2.line(img, (x, y - 26), (x, y - 8), colour, 1, cv2.LINE_AA)
            cv2.line(img, (x, y + 8), (x, y + 26), colour, 1, cv2.LINE_AA)
        cv2.rectangle(img, (0, 0), (330, 34), (0, 0, 0), -1)
        cv2.putText(img, "f%d  %s" % (t, state), (8, 24), 0, 0.62, colour, 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(args.out_dir, "%d.png" % t), img)
        written += 1
    print("wrote %d annotated frames to %s" % (written, args.out_dir))


if __name__ == "__main__":
    main()
