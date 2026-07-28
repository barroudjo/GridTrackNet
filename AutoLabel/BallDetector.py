"""Classical-CV tennis/padel ball detector used to bootstrap training labels.

Deliberately does NOT use GridTrackNet: the output of this script is meant to be
turned into ground truth for training GridTrackNet, so it has to be independent
of it.

Pipeline
  1. Background model: per-segment median of sampled frames (static camera).
  2. Play area: the blue court surface plus the cage volume above it. Anything
     outside (neighbouring courts, spectators, banners above the back wall) is
     ignored.
  3. Candidates: bright yellow-green blobs that differ from the background,
     filtered by area / aspect / fill / mean colour.
  4. Static-distractor suppression: blobs that stay in one spot for a large part
     of a window (spare balls lying on court, signage revealed when a player
     walks past) are dropped.
  5. Trajectory search: dynamic program over the candidate graph, run per
     window. A state is a pair of consecutive detections, so the transition cost
     can penalise acceleration; short gaps are allowed for occlusions. The best
     path is extracted, its candidates removed, and the search repeated.
  6. Rallies: tracks separated by less than RALLY_GAP frames are one rally.
  7. Second video pass: frames missing inside a rally are re-searched around the
     interpolated position with relaxed thresholds, and rally ends are extended
     by extrapolation.

Output: CSV with Frame,Visibility,X,Y in 1280x720 coordinates - the same
coordinate space and format LabellingTool.py writes - plus a rallies CSV.
"""
import argparse
import csv
import math
import os

import cv2
import numpy as np

WIDTH, HEIGHT = 1280, 720

# --- candidate appearance ---------------------------------------------------
HSV_LO = np.array([22, 60, 140], np.uint8)     # yellow-green, clearly bright
HSV_HI = np.array([55, 255, 255], np.uint8)
HSV_LO_RELAXED = np.array([20, 50, 125], np.uint8)
HSV_HI_RELAXED = np.array([60, 255, 255], np.uint8)
DIFF_THRESH = 20            # background difference (grayscale)
MIN_AREA, MAX_AREA = 3, 600
MAX_SIDE = 44               # a motion-blurred ball stays small
MAX_ASPECT = 4.5
MIN_FILL = 0.32             # area / bounding box
MIN_GREEN = 135             # mean G of the blob
MIN_GB = 45                 # mean G - mean B: rejects grey/dark compression noise
MIN_GR = 8                  # mean G - mean R
MIN_GREEN_RELAXED = 125
MIN_GB_RELAXED = 40
MAX_REFINE_RUN = 6          # consecutive frames the refinement may invent on its own

# --- trajectory search ------------------------------------------------------
MAX_SPEED = 105.0           # px/frame at 1280x720
MAX_GAP = 3                 # frames a track may skip
DET_REWARD = 35.0
GAP_PENALTY = 20.0          # per skipped frame
ACC_CAP = 95.0              # robust ceiling on the acceleration penalty
ACC_WEIGHT = 1.0
MIN_TRACK_LEN = 4
MIN_TRACK_SPEED = 1.5       # px/frame, rejects tracks locked onto static blobs
MIN_TRACK_TRAVEL = 35.0     # px, bounding box diagonal of the whole track

# --- rallies ----------------------------------------------------------------
RALLY_GAP = 25              # frames without ball that still count as the same rally
MIN_RALLY_FRAMES = 45       # 1.5 s at 30 fps
MIN_RALLY_DETS = 25
RALLY_EXTEND = 10           # frames of extrapolation at each end


def build_background(video, start, count, samples=32):
    """Median frame over `samples` frames spread across [start, start+count)."""
    cap = cv2.VideoCapture(video)
    idx = np.linspace(start, start + max(1, count) - 1, samples).astype(int)
    stack = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            stack.append(cv2.resize(frame, (WIDTH, HEIGHT)))
    cap.release()
    if not stack:
        return None
    return np.median(np.stack(stack), axis=0).astype(np.uint8)


def blob_candidates(mask, frame_bgr=None, offset=(0, 0), relaxed=False):
    """Connected components of `mask` that look like a ball.

    When `frame_bgr` is given the blob's mean colour is checked too, which is
    what separates a real ball from compression noise on dark fencing.
    """
    min_green = MIN_GREEN_RELAXED if relaxed else MIN_GREEN
    min_gb = MIN_GB_RELAXED if relaxed else MIN_GB
    out = []
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if area < MIN_AREA or area > MAX_AREA or max(w, h) > MAX_SIDE:
            continue
        if max(w, h) / max(1, min(w, h)) > MAX_ASPECT:
            continue
        if area / max(1, w * h) < MIN_FILL:
            continue
        if frame_bgr is not None:
            b, g, r = cv2.mean(frame_bgr, (lab == i).astype(np.uint8))[:3]
            if g < min_green or g - b < min_gb or g - r < MIN_GR:
                continue
        out.append((float(cent[i][0]) + offset[0], float(cent[i][1]) + offset[1], int(area)))
    return out


def build_play_mask(bg):
    """Region where a ball can plausibly be: the court surface plus the air above it."""
    hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, np.array([90, 90, 50], np.uint8), np.array([125, 255, 255], np.uint8))
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    # the painted service lines cut the surface into pieces - bridge them
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(blue, 8)
    if n < 2:
        return np.full((HEIGHT, WIDTH), 255, np.uint8)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 0.02 * WIDTH * HEIGHT]
    if not keep:
        keep = [1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))]
    court = np.isin(lab, keep).astype(np.uint8) * 255
    hull = cv2.convexHull(cv2.findNonZero(court))
    mask = np.zeros((HEIGHT, WIDTH), np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    # sideways margin for the glass walls, and headroom for lobs
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 41)))
    rows = np.nonzero(mask.any(axis=1))[0]
    headroom = min(90, int(0.15 * (rows[-1] - rows[0])))
    shifted = np.zeros_like(mask)
    shifted[:HEIGHT - headroom] = mask[headroom:]
    return cv2.bitwise_or(mask, shifted)


def frame_candidates(frame_bgr, bg_gray, bg_yellow, play_mask=None):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)
    color = cv2.inRange(hsv, HSV_LO, HSV_HI)
    color[bg_yellow > 0] = 0
    if play_mask is not None:
        color = cv2.bitwise_and(color, play_mask)
    motion = (np.abs(gray - bg_gray) > DIFF_THRESH).astype(np.uint8) * 255
    motion = cv2.dilate(motion, np.ones((9, 9), np.uint8))
    mask = cv2.bitwise_and(color, motion)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return blob_candidates(mask, frame_bgr)


def local_search(frame, pred, radius, bg_yellow, play_mask, static_mask=None):
    """Relaxed colour-only search for the ball around a predicted position."""
    x0 = max(0, int(pred[0] - radius)); x1 = min(WIDTH, int(pred[0] + radius))
    y0 = max(0, int(pred[1] - radius)); y1 = min(HEIGHT, int(pred[1] + radius))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    roi = frame[y0:y1, x0:x1]
    mask = cv2.inRange(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV), HSV_LO_RELAXED, HSV_HI_RELAXED)
    mask[bg_yellow[y0:y1, x0:x1] > 0] = 0
    mask[play_mask[y0:y1, x0:x1] == 0] = 0
    if static_mask is not None:
        mask[static_mask[y0:y1, x0:x1] > 0] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cand = blob_candidates(mask, roi, offset=(x0, y0), relaxed=True)
    if not cand:
        return None
    cand.sort(key=lambda c: math.hypot(c[0] - pred[0], c[1] - pred[1]))
    return cand[0][0], cand[0][1]


def drop_frozen(det, radius=6.0, min_span=20):
    """Delete detections that sit still - a ball at rest is not in play.

    Also removes small static artefacts that survived the blob-level
    suppression, which otherwise anchor a rally for hundreds of frames.
    """
    ts = sorted(det)
    removed, i = 0, 0
    while i < len(ts):
        j, xs, ys = i, [det[ts[i]][0]], [det[ts[i]][1]]
        while j + 1 < len(ts):
            x, y = det[ts[j + 1]]
            if max(max(xs), x) - min(min(xs), x) > radius:
                break
            if max(max(ys), y) - min(min(ys), y) > radius:
                break
            xs.append(x); ys.append(y); j += 1
        if ts[j] - ts[i] + 1 >= min_span:
            for t in ts[i:j + 1]:
                del det[t]
                removed += 1
        i = j + 1
    return removed


def suppress_static(cands, frames_per_window=900, cell=8, ratio=0.12):
    """Drop candidates that keep appearing at the same spot within a window.

    Returns the filtered candidates plus, per window, a mask of the suppressed
    image regions so the refinement pass can honour the same decision.
    """
    n = len(cands)
    kept = [[] for _ in range(n)]
    masks = []
    for w0 in range(0, n, frames_per_window):
        w1 = min(n, w0 + frames_per_window)
        counts = {}
        for t in range(w0, w1):
            seen = {(int(x) // cell, int(y) // cell) for x, y, _ in cands[t]}
            for k in seen:
                counts[k] = counts.get(k, 0) + 1
        limit = max(6, min(40, int(ratio * (w1 - w0))))
        hot = {k for k, v in counts.items() if v > limit}
        mask = np.zeros((HEIGHT, WIDTH), np.uint8)
        for cx, cy in hot:
            mask[max(0, (cy - 1) * cell):(cy + 2) * cell,
                 max(0, (cx - 1) * cell):(cx + 2) * cell] = 255
        masks.append((w0, w1, mask))
        for t in range(w0, w1):
            for c in cands[t]:
                key = (int(c[0]) // cell, int(c[1]) // cell)
                if any((key[0] + dx, key[1] + dy) in hot for dx in (-1, 0, 1) for dy in (-1, 0, 1)):
                    continue
                kept[t].append(c)
    return kept, masks


def best_path(cands, alive, lo, hi):
    """Highest-scoring trajectory through the candidate graph in [lo, hi)."""
    nodes = [(t, i) for t in range(lo, hi) for i in range(len(cands[t])) if alive[t][i]]
    if len(nodes) < 2:
        return []
    by_frame = {}
    for t, i in nodes:
        by_frame.setdefault(t, []).append(i)

    edges = {}
    for t, j in nodes:
        pj = cands[t][j]
        lst = []
        for dt in range(1, MAX_GAP + 2):
            tp = t - dt
            if tp < lo:
                continue
            for i in by_frame.get(tp, ()):
                pi = cands[tp][i]
                if math.hypot(pj[0] - pi[0], pj[1] - pi[1]) <= MAX_SPEED * dt:
                    lst.append(((tp, i), GAP_PENALTY * (dt - 1)))
        edges[(t, j)] = lst

    score, back = {}, {}
    for b in sorted(nodes):
        pb = cands[b[0]][b[1]]
        for a, gapcost in edges[b]:
            pa = cands[a[0]][a[1]]
            dt_ab = b[0] - a[0]
            vabx = (pb[0] - pa[0]) / dt_ab
            vaby = (pb[1] - pa[1]) / dt_ab
            best_s, best_prev = 2 * DET_REWARD - gapcost, None
            for c, _ in edges[a]:
                key = (c, a)
                if key not in score:
                    continue
                pc = cands[c[0]][c[1]]
                dt_ca = a[0] - c[0]
                acc = math.hypot(vabx - (pa[0] - pc[0]) / dt_ca, vaby - (pa[1] - pc[1]) / dt_ca)
                s = score[key] + DET_REWARD - gapcost - ACC_WEIGHT * min(ACC_CAP, acc)
                if s > best_s:
                    best_s, best_prev = s, key
            score[(a, b)] = best_s
            back[(a, b)] = best_prev

    if not score:
        return []
    end = max(score, key=score.get)
    if score[end] <= 0:
        return []
    path = [end[1], end[0]]
    cur = back[end]
    while cur is not None:
        path.append(cur[0])
        cur = back[cur]
    path.reverse()
    return path


def path_stats(cands, path):
    pts = np.array([cands[t][i][:2] for t, i in path])
    speeds = [math.hypot(pts[k + 1][0] - pts[k][0], pts[k + 1][1] - pts[k][1]) /
              max(1, path[k + 1][0] - path[k][0]) for k in range(len(path) - 1)]
    travel = float(np.hypot(*(pts.max(axis=0) - pts.min(axis=0))))
    return (float(np.mean(speeds)) if speeds else 0.0), travel


def extract_tracks(cands, window=1800, overlap=90, max_tracks_per_window=40, verbose=True):
    """Repeated best-path extraction, run on overlapping windows for speed."""
    n = len(cands)
    alive = [[True] * len(c) for c in cands]
    tracks = []
    for w0 in range(0, n, window):
        lo, hi = max(0, w0 - overlap), min(n, w0 + window)
        for _ in range(max_tracks_per_window):
            path = best_path(cands, alive, lo, hi)
            if len(path) < MIN_TRACK_LEN:
                break
            for t, i in path:
                alive[t][i] = False
            mean_speed, travel = path_stats(cands, path)
            if mean_speed < MIN_TRACK_SPEED or travel < MIN_TRACK_TRAVEL:
                continue
            tracks.append(path)
            if verbose:
                print("  track %4d-%4d  %3d dets  %5.1f px/f"
                      % (path[0][0], path[-1][0], len(path), mean_speed))
    return tracks


def merge_tracks(cands, tracks, n):
    """One detection per frame; longer tracks win where tracks overlap."""
    det = {}
    for path in sorted(tracks, key=len):
        for t, i in path:
            det[t] = (cands[t][i][0], cands[t][i][1])
    return det


def group_rallies(det):
    """Contiguous runs of detections, allowing RALLY_GAP frames of absence."""
    if not det:
        return []
    ts = sorted(det)
    runs, cur = [], [ts[0]]
    for a, b in zip(ts, ts[1:]):
        if b - a > RALLY_GAP:
            runs.append((cur[0], cur[-1], len(cur)))
            cur = []
        cur.append(b)
    runs.append((cur[0], cur[-1], len(cur)))
    return [(s, e, c) for s, e, c in runs
            if e - s + 1 >= MIN_RALLY_FRAMES and c >= MIN_RALLY_DETS]


def plan_refinement(det, rallies, n):
    """Interpolated guesses for the frames a rally is missing: {frame: (xy, radius)}.

    These are only one of the two cues used in the refinement pass; the other is
    the running constant-velocity prediction, which is what follows a curved
    trajectory correctly.
    """
    plan = {}
    for start, end, _ in rallies:
        ts = [t for t in range(start, end + 1) if t in det]
        if len(ts) < 2:
            continue
        for t0, t1 in zip(ts, ts[1:]):
            if t1 - t0 < 2:
                continue
            p0, p1 = np.array(det[t0]), np.array(det[t1])
            step = np.hypot(*(p1 - p0)) / (t1 - t0)
            for t in range(t0 + 1, t1):
                pred = p0 + (t - t0) / (t1 - t0) * (p1 - p0)
                plan[t] = (tuple(pred), 24.0 + 0.7 * step)
        # extrapolate outwards from each end
        for anchor, other, direction in ((ts[0], ts[1], -1), (ts[-1], ts[-2], 1)):
            p_a, p_o = np.array(det[anchor]), np.array(det[other])
            v = (p_a - p_o) / float(anchor - other)
            for k in range(1, RALLY_EXTEND + 1):
                t = anchor + direction * k
                if t < 0 or t >= n or t in det:
                    continue
                pred = p_a + v * (t - anchor)
                if not (0 <= pred[0] < WIDTH and 0 <= pred[1] < HEIGHT):
                    break
                plan[t] = (tuple(pred), 26.0 + 3.0 * k)
    return plan


class RallyTracker:
    """Constant-velocity local tracker used during the refinement pass."""

    def __init__(self):
        self.t = None
        self.pos = None
        self.vel = np.zeros(2)
        self.misses = 0

    def observe(self, t, pos):
        pos = np.array(pos, dtype=float)
        if self.t is not None and 0 < t - self.t <= MAX_GAP + 2:
            self.vel = 0.5 * self.vel + 0.5 * (pos - self.pos) / (t - self.t)
        else:
            self.vel = np.zeros(2)
        self.t, self.pos, self.misses = t, pos, 0

    def predict(self, t):
        """(position, radius) or None when the state is too stale to trust."""
        if self.pos is None:
            return None
        dt = t - self.t
        if dt <= 0 or dt > MAX_GAP + 6:
            return None
        pred = self.pos + self.vel * dt
        if not (-40 <= pred[0] < WIDTH + 40 and -40 <= pred[1] < HEIGHT + 40):
            return None
        return pred, min(70.0, 20.0 + 0.35 * float(np.hypot(*self.vel)) + 7.0 * dt)


def main():
    ap = argparse.ArgumentParser(description="Classical-CV ball detector (GridTrackNet-free)")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--rallies_csv", default="")
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--num_frames", type=int, default=0, help="0 = until end of video")
    ap.add_argument("--segment", type=int, default=1800, help="frames per background segment")
    ap.add_argument("--refine_passes", type=int, default=2)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit("cannot open %s" % args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_req = args.num_frames if args.num_frames > 0 else (
        total - args.start_frame if total > 0 else 10 ** 9)
    seg_len = args.segment

    def seg_models(seg):
        bg = build_background(args.video, args.start_frame + seg * seg_len, seg_len)
        by = cv2.inRange(cv2.cvtColor(bg, cv2.COLOR_BGR2HSV), HSV_LO, HSV_HI)
        return (cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).astype(np.int16),
                cv2.dilate(by, np.ones((7, 7), np.uint8)),
                build_play_mask(bg))

    print("Pass 1: candidate extraction")
    models = {}
    cands = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    for t in range(n_req):
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        seg = t // seg_len
        if seg not in models:
            models[seg] = seg_models(seg)
            print("  background model for segment %d (frame %d)" % (seg, t))
        cands.append(frame_candidates(frame, *models[seg]))
    cap.release()
    n = len(cands)
    print("  %d frames, %.2f candidates/frame" % (n, np.mean([len(c) for c in cands])))

    cands, static_masks = suppress_static(cands)
    print("  after static suppression: %.2f candidates/frame" % np.mean([len(c) for c in cands]))

    def static_mask_at(t):
        for w0, w1, m in static_masks:
            if w0 <= t < w1:
                return m
        return static_masks[-1][2]

    strong = set()      # frames whose detection comes from pass 1, not refinement

    print("Trajectory search")
    tracks = extract_tracks(cands, verbose=False)
    det = merge_tracks(cands, tracks, n)
    frozen = drop_frozen(det)
    strong.update(det)
    rallies = group_rallies(det)
    print("  %d tracks -> %d detections (%d frozen dropped) -> %d rallies"
          % (len(tracks), len(det), frozen, len(rallies)))

    for it in range(args.refine_passes):
        plan = plan_refinement(det, rallies, n)
        spans = {}
        for s, e, _ in rallies:
            for t in range(max(0, s - RALLY_EXTEND), min(n, e + RALLY_EXTEND + 1)):
                spans[t] = None
        print("Pass %d: refining inside %d rally frames" % (it + 2, len(spans)))
        cap = cv2.VideoCapture(args.video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
        tracker = RallyTracker()
        prev_t, found, run = None, 0, 0
        for t in range(n):
            ok, frame = cap.read()
            if not ok:
                break
            if t not in spans:
                continue
            if prev_t is None or t - prev_t > 1:
                tracker = RallyTracker()      # new rally, drop the old state
                run = 0
            prev_t = t
            if t in det:
                tracker.observe(t, det[t])
                if t in strong:
                    run = 0
                continue
            if run >= MAX_REFINE_RUN:
                continue
            guesses = [g for g in (tracker.predict(t), plan.get(t)) if g is not None]
            if not guesses:
                continue
            frame = cv2.resize(frame, (WIDTH, HEIGHT))
            _, bg_yellow, play_mask = models[min(max(models), t // seg_len)]
            smask = static_mask_at(t)
            best, best_d = None, 1e9
            for pred, radius in guesses:
                hit = local_search(frame, pred, radius, bg_yellow, play_mask, smask)
                if hit:
                    d = math.hypot(hit[0] - pred[0], hit[1] - pred[1])
                    if d < best_d:
                        best, best_d = hit, d
            if best:
                det[t] = best
                tracker.observe(t, best)
                found += 1
                run += 1
        cap.release()
        drop_frozen(det)
        print("  recovered %d frames" % found)
        rallies = group_rallies(det)
        if not found:
            break

    rallies = group_rallies(det)
    covered = sum(1 for s, e, _ in rallies for t in range(s, e + 1) if t in det)
    span = sum(e - s + 1 for s, e, _ in rallies)
    print("  %d rallies, %d frames, ball located in %d (%.1f%%)"
          % (len(rallies), span, covered, 100.0 * covered / max(1, span)))

    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Frame", "Visibility", "X", "Y"])
        for t in range(n):
            if t in det:
                w.writerow([t, 1, int(round(det[t][0])), int(round(det[t][1]))])
            else:
                w.writerow([t, 0, 0, 0])
    print("Wrote %s" % args.out_csv)

    rallies_csv = args.rallies_csv or os.path.splitext(args.out_csv)[0] + "_rallies.csv"
    with open(rallies_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Rally", "StartFrame", "EndFrame", "Frames", "Detections"])
        for k, (s, e, _) in enumerate(rallies, 1):
            c = sum(1 for t in range(s, e + 1) if t in det)
            w.writerow([k, s + args.start_frame, e + args.start_frame, e - s + 1, c])
    print("Wrote %s" % rallies_csv)


if __name__ == "__main__":
    main()
