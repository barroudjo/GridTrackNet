"""Compare two sets of ball labels.

Reports the numbers that matter for GridTrackNet training: how often the two
runs agree that a ball is visible at all, how far apart they place it, and
whether they land in the same 48x27 output grid cell (26.7 x 26.7 px).
"""
import argparse
import csv
import json
import math
import os

WIDTH, HEIGHT = 1280, 720
GRID_COLS, GRID_ROWS = 48, 27
CELL_W, CELL_H = WIDTH / GRID_COLS, HEIGHT / GRID_ROWS
TOL = 4.0          # the paper's evaluation tolerance, in px


def load(path):
    """-> {frame: (x, y) or None}; missing frames are simply absent."""
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


def cell(p):
    return int(p[0] // CELL_W), int(p[1] // CELL_H)


def pct(n, d):
    return "%5.1f%%" % (100.0 * n / d) if d else "    -"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, help="labels treated as ground truth")
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=10 ** 9)
    ap.add_argument("--worst", type=int, default=10, help="list this many largest errors")
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    ref, cand = load(args.reference), load(args.candidate)
    frames = [t for t in range(args.start, args.end + 1) if t in ref and t in cand]
    missing = [t for t in range(args.start, args.end + 1) if t in ref and t not in cand]

    both_vis, both_none, ref_only, cand_only = [], [], [], []
    for t in frames:
        r, c = ref[t], cand[t]
        if r and c:
            both_vis.append(t)
        elif not r and not c:
            both_none.append(t)
        elif r:
            ref_only.append(t)
        else:
            cand_only.append(t)

    errs = {t: math.hypot(ref[t][0] - cand[t][0], ref[t][1] - cand[t][1]) for t in both_vis}
    same_cell = [t for t in both_vis if cell(ref[t]) == cell(cand[t])]
    ordered = sorted(errs, key=errs.get)
    vals = [errs[t] for t in ordered]

    def q(f):
        return vals[min(len(vals) - 1, int(f * len(vals)))] if vals else float("nan")

    n = len(frames)
    print("Compared %d frames in [%d, %d]%s"
          % (n, args.start, args.end,
             ("   (%d reference frames missing from candidate)" % len(missing)) if missing else ""))
    print()
    print("VISIBILITY AGREEMENT")
    print("  both say visible        %4d  %s" % (len(both_vis), pct(len(both_vis), n)))
    print("  both say no ball        %4d  %s" % (len(both_none), pct(len(both_none), n)))
    print("  reference only          %4d  %s   (candidate missed a visible ball)"
          % (len(ref_only), pct(len(ref_only), n)))
    print("  candidate only          %4d  %s   (candidate labelled a ball that isn't there,"
          % (len(cand_only), pct(len(cand_only), n)))
    print("                                       or found one the reference missed)")
    print("  agreement               %s" % pct(len(both_vis) + len(both_none), n))
    print()
    if vals:
        print("POSITION ERROR on the %d frames both call visible (px)" % len(both_vis))
        print("  median %.1f    mean %.1f    p90 %.1f    max %.1f"
              % (q(0.5), sum(vals) / len(vals), q(0.9), vals[-1]))
        print("  within %.0f px (paper tolerance)   %4d  %s"
              % (TOL, sum(1 for v in vals if v <= TOL), pct(sum(1 for v in vals if v <= TOL), len(vals))))
        print("  within 10 px                      %4d  %s"
              % (sum(1 for v in vals if v <= 10), pct(sum(1 for v in vals if v <= 10), len(vals))))
        print("  same 48x27 grid cell              %4d  %s"
              % (len(same_cell), pct(len(same_cell), len(both_vis))))
        print()
        print("LARGEST DISAGREEMENTS")
        for t in ordered[-args.worst:][::-1]:
            print("  f%-4d ref (%4.0f,%4.0f)  cand (%4.0f,%4.0f)  %6.1f px"
                  % (t, ref[t][0], ref[t][1], cand[t][0], cand[t][1], errs[t]))
    if ref_only:
        print("\n  frames the candidate called 'none' but reference has a ball: %s" % ref_only)
    if cand_only:
        print("  frames the candidate placed a ball but reference says 'none': %s" % cand_only)

    if args.out_json:
        json.dump({
            "frames": n, "both_visible": len(both_vis), "both_none": len(both_none),
            "reference_only": ref_only, "candidate_only": cand_only,
            "median_px": q(0.5) if vals else None,
            "mean_px": (sum(vals) / len(vals)) if vals else None,
            "p90_px": q(0.9) if vals else None,
            "max_px": vals[-1] if vals else None,
            "within_tol": sum(1 for v in vals if v <= TOL),
            "same_cell": len(same_cell),
            "errors": {str(t): errs[t] for t in errs},
        }, open(args.out_json, "w"), indent=1)
        print("\nwrote %s" % args.out_json)


if __name__ == "__main__":
    main()
