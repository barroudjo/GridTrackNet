"""Incremental store for hand-read (model-read) ball positions.

Labels are kept in a JSON file as {frame: [x, y]} for visible balls and
{frame: null} for frames explicitly judged to have no ball in play, so that
"not looked at yet" stays distinguishable from "looked at, nothing there".

Usage
    python LabelStore.py put  store.json "300=524,246 301=530,250 302=none"
    python LabelStore.py show store.json --start 290 --end 320
    python LabelStore.py csv  store.json Labels.csv --start 246 --end 448
"""
import argparse
import csv
import json
import os


def load(path):
    if os.path.exists(path):
        return json.load(open(path))
    return {}


def save(path, data):
    json.dump(data, open(path, "w"), indent=0, sort_keys=True)


def cmd_put(args):
    data = load(args.store)
    n = 0
    for item in args.values.split():
        frame, _, pos = item.partition("=")
        if pos.lower() in ("none", "n", "-"):
            data[str(int(frame))] = None
        else:
            x, y = pos.split(",")
            data[str(int(frame))] = [float(x), float(y)]
        n += 1
    save(args.store, data)
    print("stored %d frames, %d total" % (n, len(data)))


def cmd_show(args):
    data = load(args.store)
    ks = sorted(int(k) for k in data)
    ks = [k for k in ks if args.start <= k <= args.end]
    miss = [k for k in range(args.start, min(args.end, max(ks) if ks else args.end) + 1)
            if str(k) not in data]
    print("labelled %d frames in [%d, %d]" % (len(ks), args.start, args.end))
    print("visible: %d   none: %d" % (sum(1 for k in ks if data[str(k)]),
                                      sum(1 for k in ks if not data[str(k)])))
    if miss:
        print("unlabelled: %s%s" % (miss[:40], " ..." if len(miss) > 40 else ""))
    for k in ks:
        v = data[str(k)]
        print("  %4d  %s" % (k, "none" if v is None else "%.0f,%.0f" % (v[0], v[1])))


def cmd_csv(args):
    data = load(args.store)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Frame", "Visibility", "X", "Y"])
        for i, t in enumerate(range(args.start, args.end + 1)):
            v = data.get(str(t))
            idx = i if args.reindex else t
            if v:
                w.writerow([idx, 1, int(round(v[0])), int(round(v[1]))])
            else:
                w.writerow([idx, 0, 0, 0])
    print("wrote %s (%d rows)" % (args.out, args.end - args.start + 1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("put"); p.add_argument("store"); p.add_argument("values"); p.set_defaults(fn=cmd_put)
    p = sub.add_parser("show"); p.add_argument("store")
    p.add_argument("--start", type=int, default=0); p.add_argument("--end", type=int, default=10 ** 9)
    p.set_defaults(fn=cmd_show)
    p = sub.add_parser("csv"); p.add_argument("store"); p.add_argument("out")
    p.add_argument("--start", type=int, required=True); p.add_argument("--end", type=int, required=True)
    p.add_argument("--reindex", action="store_true", help="renumber frames from 0")
    p.set_defaults(fn=cmd_csv)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
