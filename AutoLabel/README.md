# AutoLabel — building GridTrackNet training data without GridTrackNet

Tooling and findings from an experiment: **can Claude build a GridTrackNet
training set by looking at the video frames itself?**

The labels are ground truth for training GridTrackNet, so they must not come
from GridTrackNet. Two approaches were tried on
`Sample Video's/video9_cut_2.mp4` (900 frames, 30 fps, 1920x1080, padel, fixed
camera):

1. A classical-CV detector — [`BallDetector.py`](BallDetector.py). Works, but
   is a pile of hand-tuned thresholds and gets the *semantic* calls wrong.
2. **Claude reading ball positions off the frames directly** — the approach
   this document is mostly about.

Short answer: (2) works and produces a valid dataset end to end, and it is the
only one of the two that reliably answers "which of the two balls on court is
in play". It is far too slow to be the bulk labeller for a full match.

---

## Result

The full README pipeline ran end to end from hand-read labels:

| Step | Output |
|---|---|
| `FrameGenerator.py` on the source clip | 900 frames @ 1280x720 |
| Visual labelling of frames 24–95 | 53 visible, 19 no-ball-in-play |
| Frame-exact trim to the rally | `Dataset/clips/rally1.mp4`, 72 frames |
| `FrameGenerator.py` + `Labels.csv` | `Dataset/match1/` |
| `DataGen.py` | 61 train + 7 val instances |

Verified by decoding a TFRecord back: instance 1 yields
`(872,357) (881,358) (880,362) (893,368) (895,376)` — exactly the labels for
those frames, in the correct 48x27 grid cells with correct offsets.

`Dataset/` is gitignored (~3 GB). Regenerate it with the commands in
[Reproducing](#reproducing).

---

## The method that works

Three things mattered, in order of impact.

### 1. Motion is what disambiguates — a single still frame is not enough

The back wall carries a painted `⊙` logo that is indistinguishable from a ball
in a still frame. `FrameView.py --motion` tints pixels that changed against the
neighbouring frames: static clutter stays untinted, the ball is a small moving
blob. Without it I confidently "found" a ball at (700,140) that was the logo.

This is a *display* transform, not a detector — it shows what moved; the
decision about which moving thing is the ball stays with the model.

### 2. A full-span crosshair, not a fraction of the tile

Estimating a position as "about 40% across the tile" gives **±20 px** and
produces physically impossible zigzag trajectories. Rendering a full-span
crosshair at a known point, with tick marks every 10 original px, and reading
the ball as an *offset from that line*, gives **±3 px**.

### 3. Coarse locate, then fine confirm

- **Coarse:** 9 frames per montage, `--crop 200-400 --scale 1.8-3.0 --grid 25
  --motion`, crosshair at a fixed point. Gets within ~±20 px.
- **Fine:** same 9 frames, `--crop 72-96 --scale 5-6 --grid 10 --motion`, with
  `--centers` *and* `--marks` set to the coarse estimates.

That is **~2 image reads per 9 frames**. The trajectory is the error check: a
ball in flight changes velocity gradually, so a kink means a misread.

### What it caught that the CV pipeline did not

- **Two balls are on court for most of the clip.** One comes to rest at
  ~(524,246) around f115 and stays there. The CV tracker anchored rallies to it.
- A **phantom static blob at (558,228)** held a "rally" open for 100+ frames.
- **Frames 240–450 are not a rally** — one far player is knocking a ball about
  while the others stand still. The CV pipeline called it a 200-frame rally.
- **Frames 44–58: the ball is genuinely out of frame** below the bottom edge,
  with all four players watching. Correctly `INVISIBLE`, not a miss.

---

## Model comparison: Opus 5 vs Sonnet 5 (medium effort)

Same 72 frames, same tools, same method prompt; the Sonnet run was blind to the
Opus labels.

| | Opus 5 | Sonnet 5 (medium) |
|---|---|---|
| Montages / turns | 31 / 62 | 40 / 88 |
| Median image size | 2,458 tok | 4,424 tok |
| Tokens (fresh + output) | ~171 k | **276 k** (measured) |
| Wall clock | — | 34 min |

**Agreement**

| Metric | Value |
|---|---|
| Visibility agreement | **98.6%** (53 both-visible, 18 both-none, 1 disagreement) |
| Median position error | 17.9 px (p90 76, max 118) |
| Within 4 px (paper tolerance) | 7.5% |
| Same 48x27 grid cell | 35.8% |
| Frames >40 px apart | **13 of 53**, clustered at f77–f87, offset ~70 px right |

Sonnet reliably decides *whether* a ball is in play. It does not reliably
decide *where*. The failure is horizontal registration — median `|dx|` and
`|dy|` are both ~8 px, but max `|dx|` is 109 px vs 91 px for y.

**Evidence this is Sonnet and not just disagreement**

- Four disputed frames (63, 78, 84, 87) re-rendered on fresh crops with
  different centres: the ball sits unambiguously on the Opus coordinates. The
  gaps (73–118 px) are far beyond reading noise.
- **Trajectory smoothness, which needs no ground truth at all** — frame-to-frame
  acceleration of each track:

  | | median | p90 | max |
  |---|---|---|---|
  | Opus | 8.1 px/f² | 15.7 | 36 |
  | Sonnet | 18.7 px/f² | 88.1 | 135 |

  A ball in flight cannot accelerate 88 px/f² between frames. The Sonnet track
  is non-physical on its own terms.

At 36% grid-cell agreement those labels would teach the model a systematic
horizontal bias. **The tier difference buys accuracy, not polish.**

---

## Cost

Both runs are dominated by images and by re-sending the growing context each
turn. Sonnet used ~2x the tokens at ~0.4x the price, so the saving is far
smaller than the sticker difference.

| | 72 frames, single context | context chunked every 12 turns |
|---|---|---|
| Opus 5 ($5 / $25 per M) | $5.13 | $3.04 |
| Sonnet 5 ($2 / $10 intro) | $3.62 | **$1.73** |

**Per hour of processed video**, Sonnet, chunked — the cheapest honest config:

| Scope | Tokens | Cost |
|---|---|---|
| Every frame (108,000) | 4.5 B | ~$2,590 |
| Rally frames only (~35%) | 1.6 B | ~$910 |

A heavily tuned pipeline (9 frames per montage, ~1,500-token images) reaches
~230 M tokens/hour ≈ $460 — but 9 frames per montage is already past the point
where Sonnet's accuracy broke down.

One hour of video is **~1.6 billion tokens in the best case**. That is orders of
magnitude beyond any Claude Code weekly allowance.

Re-measure with [`RunCost.py`](RunCost.py); the model is stated in that file's
docstring and every assumption is a flag.

---

## Recommendation

Direct visual labelling is the right tool for the calls that need judgement,
and the wrong tool for bulk coordinates:

- **Use it for:** rally boundaries, multi-ball disambiguation, deciding whether
  a stretch is play or dead time, and auditing a sample of any automated
  labeller. Sonnet is adequate here (98.6% on the visibility call) and cheap.
- **Do not use it for:** every frame of a match. The per-frame cost has to fall
  ~100x first.

The practical shape is a cheap detector for coordinates plus Claude for the
semantics and the audit — not one or the other.

---

## Tools

| Script | Purpose |
|---|---|
| [`FrameView.py`](FrameView.py) | Montage renderer: coordinate grid in original-frame coords, `--motion` tint, full-span `--marks` crosshair with 10 px ticks, per-frame `--centers`. The core labelling tool. |
| [`LabelStore.py`](LabelStore.py) | Incremental label store. Keeps "looked at, no ball" distinct from "not looked at yet". Exports `Labels.csv` with `--reindex`. |
| [`DrawLabels.py`](DrawLabels.py) | Draws labels onto frames — marker, recent-position trail, and a VISIBLE / no-ball / UNLABELLED banner. |
| [`VerifySheet.py`](VerifySheet.py) | Two contact sheets: crops centred on `VISIBLE` labels (the cross should be on the ball) and on the interpolated position of `INVISIBLE` ones (these should show no ball). |
| [`CompareLabels.py`](CompareLabels.py) | Compares two label sets: visibility agreement, position error, 48x27 grid-cell agreement, largest disagreements. |
| [`RunCost.py`](RunCost.py) | Token/cost model for a labelling run from the montages it rendered. |
| [`BallDetector.py`](BallDetector.py) | The classical-CV detector. Kept for reference — see the caveat below. |
| [`ffmpeg_util.py`](ffmpeg_util.py) | Locates a working ffmpeg/ffprobe. |

### Data files

- `video9_cut_2_claude_labels.json` — the Opus 5 labels (frames 24–95).
- `video9_cut_2_sonnet_labels.json` — the Sonnet 5 labels for the same frames.
- `opus_vs_sonnet_compare.json` — per-frame error dump from `CompareLabels.py`.

### Caveat on `BallDetector.py`

It runs and it finds the ball, but it is a stack of hand-tuned constants (HSV
bounds, blob area/fill/aspect limits, mean-colour gates, static-suppression
ratios, DP reward/penalty weights) fitted to this one court under this one
lighting. It is brittle by construction, and independently of tuning it makes
the wrong *semantic* calls listed above. Treat it as a candidate generator to
be audited, not a labeller.

### Environment note

There is no `ffmpeg`/`ffprobe` on PATH on this machine and no `imageio-ffmpeg`
or `PyAV` in the venv. A full ffmpeg 7 build is at
`/opt/AgentDVR/ffmpeg7/bin/`, but it needs
`LD_LIBRARY_PATH=/opt/AgentDVR/ffmpeg7/lib`. `ffmpeg_util.find_ffmpeg()` handles
this. For frame-exact cuts use
`-vf "select='between(n,A,B)',setpts=N/FRAME_RATE/TB"` rather than `-ss`/`-t`.

---

## Reproducing

```bash
# 1. Extract all frames of the source clip
.venv/bin/python FrameGenerator.py \
    --video_dir "Sample Video's/video9_cut_2.mp4" \
    --export_dir Dataset/_work/video9_full

# 2. Label visually — coarse locate, then fine confirm
.venv/bin/python AutoLabel/FrameView.py \
    --frames_dir Dataset/_work/video9_full/frames --out /tmp/coarse.png \
    --start 59 --count 9 --cols 3 --crop 300 --scale 1.8 \
    --center 690,650 --marks '{"59":[690,650], ...}' --grid 25 --motion

.venv/bin/python AutoLabel/FrameView.py \
    --frames_dir Dataset/_work/video9_full/frames --out /tmp/fine.png \
    --frames 59,60,61,62,63,64,65,66,67 --cols 3 --crop 72 --scale 6 \
    --centers '{"59":[676,716], ...}' --marks '{"59":[676,716], ...}' \
    --grid 10 --motion

.venv/bin/python AutoLabel/LabelStore.py put labels.json "59=699,716 60=688,701"

# 3. Trim to the rally (frame-exact) and rebuild the match folder
export LD_LIBRARY_PATH=/opt/AgentDVR/ffmpeg7/lib
/opt/AgentDVR/ffmpeg7/bin/ffmpeg -y -i "Sample Video's/video9_cut_2.mp4" \
    -vf "select='between(n,24,95)',setpts=N/FRAME_RATE/TB" -an \
    -c:v libx264 -crf 16 Dataset/clips/rally1.mp4

.venv/bin/python FrameGenerator.py \
    --video_dir Dataset/clips/rally1.mp4 --export_dir Dataset/match1
.venv/bin/python AutoLabel/LabelStore.py csv \
    AutoLabel/video9_cut_2_claude_labels.json Dataset/match1/Labels.csv \
    --start 24 --end 95 --reindex

# 4. Generate TFRecords
.venv/bin/python DataGen.py --input_dir Dataset \
    --export_dir Dataset/TFRecordFiles --val_split 0.2 \
    --augment_data 1 --next_img_index 2

# 5. Audit
.venv/bin/python AutoLabel/DrawLabels.py \
    --frames_dir Dataset/_work/video9_full/frames \
    --labels AutoLabel/video9_cut_2_claude_labels.json \
    --out_dir Dataset/_work/video9_full/frames_annotated --start 24 --end 95
.venv/bin/python AutoLabel/CompareLabels.py \
    --reference AutoLabel/video9_cut_2_claude_labels.json \
    --candidate AutoLabel/video9_cut_2_sonnet_labels.json --start 24 --end 95
```

---

## Scope

Frames 24–95 of one clip were labelled — a contiguous ball flight with two
bounces and a hit, enough to validate the method and exercise the whole
pipeline. It is **not** a training set on its own. The 30-minute
`sysffmpeg__*.mkv` has not been touched.
