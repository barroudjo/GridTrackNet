"""Estimate the token cost of a visual-labelling run.

Two parts:
  1. Measured   - the montages actually rendered, and their image-token cost.
  2. Modelled   - the conversation cost, which is dominated by re-sending the
                  growing context each turn. With prompt caching most of that
                  is billed at the cache-read rate.

Image tokens follow Anthropic's formula, width * height / 750, after the image
is scaled down to the model's long-edge limit (2576 px on the high-resolution
tier: Opus 4.7+ and Sonnet 5; 1568 px before that).
"""
import argparse
import glob
import os

import cv2

LONG_EDGE = {"highres": 2576, "legacy": 1568}
TOKENS_PER_PX = 1.0 / 750.0
MAX_IMAGE_TOKENS = {"highres": 4784, "legacy": 1600}

PRICES = {                       # USD per million tokens (input, output)
    "sonnet-5": (3.00, 15.00),
    "sonnet-5-intro": (2.00, 10.00),
    "opus-5": (5.00, 25.00),
}
CACHE_READ = 0.1
CACHE_WRITE = 1.25


def image_tokens(w, h, tier="highres"):
    limit = LONG_EDGE[tier]
    if max(w, h) > limit:
        s = limit / float(max(w, h))
        w, h = w * s, h * s
    return min(MAX_IMAGE_TOKENS[tier], int(round(w * h * TOKENS_PER_PX)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views_dir", required=True, help="dir of rendered montages")
    ap.add_argument("--frames_labelled", type=int, required=True)
    ap.add_argument("--tier", default="highres", choices=list(LONG_EDGE))
    ap.add_argument("--model", default="sonnet-5-intro", choices=list(PRICES))
    ap.add_argument("--base_context", type=int, default=20000,
                    help="system prompt + tools + task, resent every turn")
    ap.add_argument("--text_per_turn", type=int, default=600,
                    help="tool call + tool text output added per turn")
    ap.add_argument("--output_per_turn", type=int, default=800,
                    help="reasoning + tool call emitted per turn")
    ap.add_argument("--chunk_turns", type=int, default=0,
                    help="restart context every N turns (0 = never; quadratic growth)")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.views_dir, "*.png")) +
                   glob.glob(os.path.join(args.views_dir, "*.jpg")))
    if not paths:
        raise SystemExit("no montages found in %s" % args.views_dir)

    sizes, toks = [], []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        sizes.append((w, h))
        toks.append(image_tokens(w, h, args.tier))

    n_img = len(toks)
    print("MEASURED")
    print("  montages rendered        %d" % n_img)
    print("  frames labelled          %d  (%.1f frames per montage)"
          % (args.frames_labelled, args.frames_labelled / float(n_img)))
    print("  image tokens each        min %d  median %d  max %d"
          % (min(toks), sorted(toks)[len(toks) // 2], max(toks)))
    print("  image tokens total       %s" % format(sum(toks), ","))
    print()

    # one turn per montage: render call + read call. The read carries the image.
    turns = n_img * 2
    per_turn_new = [0] * turns
    for i, t in enumerate(toks):
        per_turn_new[2 * i] = args.text_per_turn          # the render tool call
        per_turn_new[2 * i + 1] = t + args.text_per_turn  # the image
    chunk = args.chunk_turns or turns

    cache_read = fresh = 0
    ctx = args.base_context
    for k in range(turns):
        if k % chunk == 0:
            ctx = args.base_context
        cache_read += ctx
        fresh += per_turn_new[k]
        ctx += per_turn_new[k] + args.output_per_turn
    output = turns * args.output_per_turn

    in_price, out_price = PRICES[args.model]
    cost = (cache_read * CACHE_READ * in_price / 1e6
            + fresh * CACHE_WRITE * in_price / 1e6
            + output * out_price / 1e6)

    print("MODELLED CONVERSATION  (%s, $%.2f/$%.2f per M, %s)"
          % (args.model, in_price, out_price,
             ("context restarts every %d turns" % chunk) if args.chunk_turns else "single context"))
    print("  turns                    %d" % turns)
    print("  cache-read input         %s tokens  ($%.2f at 0.1x)"
          % (format(cache_read, ","), cache_read * CACHE_READ * in_price / 1e6))
    print("  fresh input              %s tokens  ($%.2f at 1.25x)"
          % (format(fresh, ","), fresh * CACHE_WRITE * in_price / 1e6))
    print("  output                   %s tokens  ($%.2f)"
          % (format(output, ","), output * out_price / 1e6))
    print("  TOTAL                    $%.2f   ($%.4f per labelled frame)"
          % (cost, cost / args.frames_labelled))
    print()

    per_frame = cost / args.frames_labelled
    for label, frames in (("1 minute of 30fps video", 1800),
                          ("1 hour of 30fps video", 108000),
                          ("1 hour, rally frames only (35%)", 37800)):
        print("  %-34s %8s frames   $%9.2f" % (label, format(frames, ","), per_frame * frames))


if __name__ == "__main__":
    main()
