"""Render montages of extracted frames for visual (model-in-the-loop) labelling.

Each tile carries a coordinate grid labelled in ORIGINAL 1280x720 frame
coordinates, so a ball spotted in a tile can be read off directly as an (x, y)
pair - no mental arithmetic about crop offsets.

Examples
    # six full frames, 2 per row
    python FrameView.py --frames_dir Dataset/match1/frames --out /tmp/a.png \
        --start 300 --count 6 --cols 3 --crop 0 --scale 0.5

    # twelve consecutive frames, 384px crops following a constant-velocity guess
    python FrameView.py --frames_dir Dataset/match1/frames --out /tmp/b.png \
        --start 300 --count 12 --cols 4 --crop 384 --center 520,240 --velocity -8,3
"""
import argparse
import json
import os

import cv2
import numpy as np

WIDTH, HEIGHT = 1280, 720


def draw_grid(tile, ox, oy, scale, step, alpha=0.45):
    """Overlay a faint coordinate grid; ox/oy are the original coords of tile (0,0)."""
    h, w = tile.shape[:2]
    over = tile.copy()
    for xo in range(int(np.ceil(ox / step) * step), int(ox + w / scale) + 1, step):
        xp = int((xo - ox) * scale)
        cv2.line(over, (xp, 0), (xp, h), (0, 220, 255), 1)
        cv2.putText(over, str(xo), (xp + 2, 12), 0, 0.38, (0, 220, 255), 1, cv2.LINE_AA)
    for yo in range(int(np.ceil(oy / step) * step), int(oy + h / scale) + 1, step):
        yp = int((yo - oy) * scale)
        cv2.line(over, (0, yp), (w, yp), (0, 220, 255), 1)
        cv2.putText(over, str(yo), (2, yp - 3), 0, 0.38, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.addWeighted(over, alpha, tile, 1 - alpha, 0, tile)


def add_motion_tint(frame, prev, nxt, thresh=16):
    """Tint pixels that changed since the previous/next frame.

    A display aid only: it shows what moved, it does not decide what the ball
    is. Static clutter (logos, line markings, banners) stays untinted, which is
    what makes a small ball distinguishable from a painted circle.
    """
    if prev is None and nxt is None:
        return frame
    moved = np.zeros(frame.shape[:2], np.uint8)
    for other in (prev, nxt):
        if other is None:
            continue
        d = cv2.absdiff(frame, other).max(axis=2)
        moved = np.maximum(moved, (d > thresh).astype(np.uint8) * 255)
    moved = cv2.dilate(moved, np.ones((3, 3), np.uint8))
    out = frame.copy()
    tint = out.astype(np.int16)
    tint[..., 2] = np.where(moved > 0, np.minimum(255, tint[..., 2] + 90), tint[..., 2])
    tint[..., 0] = np.where(moved > 0, np.minimum(255, tint[..., 0] + 60), tint[..., 0])
    return tint.astype(np.uint8)


def make_tile(frame, index, crop, center, scale, grid_step, mark=None):
    if crop > 0:
        cx, cy = center
        x0 = int(round(cx - crop / 2.0))
        y0 = int(round(cy - crop / 2.0))
        pad = crop
        padded = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
        sub = padded[y0 + pad:y0 + pad + crop, x0 + pad:x0 + pad + crop].copy()
        ox, oy = x0, y0
    else:
        sub = frame.copy()
        ox, oy = 0, 0
    if scale != 1.0:
        sub = cv2.resize(sub, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_NEAREST if scale > 1 else cv2.INTER_AREA)
    if mark is not None:
        # full-span crosshair: judging "left or right of the line" is far more
        # reliable than estimating a fraction of the tile width
        mx, my = int((mark[0] - ox) * scale), int((mark[1] - oy) * scale)
        h, w = sub.shape[:2]
        cv2.line(sub, (mx, 0), (mx, h), (0, 0, 255), 1)
        cv2.line(sub, (0, my), (w, my), (0, 0, 255), 1)
        for d in range(-100, 101, 10):
            tx, ty = mx + int(d * scale), my + int(d * scale)
            size = 7 if d % 50 == 0 else 4
            if 0 <= tx < w:
                cv2.line(sub, (tx, my - size), (tx, my + size), (0, 0, 255), 1)
            if 0 <= ty < h:
                cv2.line(sub, (mx - size, ty), (mx + size, ty), (0, 0, 255), 1)
        cv2.circle(sub, (mx, my), 10, (0, 0, 255), 1, cv2.LINE_AA)
    if grid_step > 0:
        draw_grid(sub, ox, oy, scale, grid_step)
    label = "f%d" % index
    cv2.rectangle(sub, (0, sub.shape[0] - 18), (74, sub.shape[0]), (0, 0, 0), -1)
    cv2.putText(sub, label, (3, sub.shape[0] - 5), 0, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(sub, (0, 0), (sub.shape[1] - 1, sub.shape[0] - 1), (255, 255, 255), 1)
    return sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--frames", default="", help="explicit comma-separated frame indices")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--crop", type=int, default=384, help="crop size in original px, 0 = full frame")
    ap.add_argument("--center", default="640,360", help="x,y of the first tile's crop centre")
    ap.add_argument("--velocity", default="0,0", help="per-step shift of the crop centre")
    ap.add_argument("--centers", default="", help="JSON {frame: [x, y]} overriding centre/velocity")
    ap.add_argument("--marks", default="", help="JSON {frame: [x, y]} drawn as a red circle")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--grid", type=int, default=100, help="grid spacing in original px, 0 = off")
    ap.add_argument("--motion", action="store_true", help="tint pixels that moved between neighbouring frames")
    args = ap.parse_args()

    if args.frames:
        indices = [int(v) for v in args.frames.split(",")]
    else:
        indices = list(range(args.start, args.start + args.count * args.step, args.step))

    cx, cy = [float(v) for v in args.center.split(",")]
    vx, vy = [float(v) for v in args.velocity.split(",")]
    centers = {int(k): v for k, v in json.loads(args.centers).items()} if args.centers else {}
    marks = {int(k): v for k, v in json.loads(args.marks).items()} if args.marks else {}

    def read(i):
        return cv2.imread(os.path.join(args.frames_dir, "%d.png" % i))

    tiles = []
    for k, idx in enumerate(indices):
        frame = read(idx)
        if frame is None:
            continue
        if args.motion:
            frame = add_motion_tint(frame, read(idx - 1), read(idx + 1))
        c = centers.get(idx, (cx + vx * k, cy + vy * k))
        tiles.append(make_tile(frame, idx, args.crop, c, args.scale,
                               args.grid, marks.get(idx)))
    if not tiles:
        raise SystemExit("no frames found")
    cols = args.cols
    blank = np.zeros_like(tiles[0])
    rows = [np.hstack(tiles[i:i + cols] + [blank] * (cols - len(tiles[i:i + cols])))
            for i in range(0, len(tiles), cols)]
    out = np.vstack(rows)
    cv2.imwrite(args.out, out)
    print("wrote %s  %dx%d  %d tiles" % (args.out, out.shape[1], out.shape[0], len(tiles)))


if __name__ == "__main__":
    main()
