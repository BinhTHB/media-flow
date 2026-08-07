# Sampling Audit

Date: 2026-08-07
Dataset: `evaluation/` (150 frames, 15 per video from 10 QA videos)

## Goals
Document how the 150 sampled frames were chosen, check for class / positive-frame
bias, and plan any corrective additions. This is a **planning record**, not a
model-quality result.

## 1. Selection method
`scripts/build_evaluation_set.py` samples frames per video using a minimum
temporal gap (default 1.0 s) plus content-variance preference, capped at
`--frames-per-video`. All sampled frames come from the 10 QA videos in
`qa_inputs/`.

- 150 frames sampled
- 15 frames per video (even split across 10 videos)
- Sampling interval: evenly spaced with ~1 s gap between candidate frames

## 2. Positive / negative balance (based on pre-labels)

Pre-labels were produced by the detector at `conf >= 0.05` and are **NOT ground
truth**. They are used here only to measure sampling bias.

| Measure | Count |
|---|---:|
| Total frames | 150 |
| Frames with >= 1 pre-label box (positive) | 138 (92%) |
| Frames with 0 boxes (negative / background) | 12 (8%) |

**Concern:** 92 % of the sampled frames already contain at least one predicted
box. This is strong **positive-frame bias** — most of the QA videos are synthetic
and were designed to continually show a subtitle / watermark / logo, and the
sampler preferred content-changing frames. A ground-truth set this positive-heavy
would under-count false positives on clean background and skew Precision.

## 3. Per-video positive/negative breakdown

| Video | frames | positive | negative |
|---|---:|---:|---:|
| qa_01_portrait_audio_subtitle | 15 | 12 | 3 |
| qa_02_highres_portrait | 15 | 13 | 2 |
| qa_03_landscape_16x9 | 15 | 8 | 7 |
| qa_04_bottom_stable | 15 | 15 | 0 |
| qa_05_moving_subtitle | 15 | 15 | 0 |
| qa_06_top_burn_in_text | 15 | 15 | 0 |
| qa_07_watermark | 15 | 15 | 0 |
| qa_08_logo_sticker | 15 | 15 | 0 |
| qa_09_ui_overlay | 15 | 15 | 0 |
| qa_10_two_line_subtitle | 15 | 15 | 0 |
| **Total** | **150** | **138** | **12** |

Videos qa_04 .. qa_10 are object-dense by design, so they contribute almost no
negative frames. Only qa_01 and qa_03 contain regions without a predicted object.

## 4. Class distribution of pre-label boxes

| Class | pre-label box count |
|---|---:|
| subtitle | 31 |
| other_burn_in_text | 168 |
| watermark | 11 |
| logo | 1 |

The `other_burn_in_text` class dominates (74%). Because these are machine
pre-labels, this partly reflects the model over-predicting `other_burn_in_text`
on textures / glyph fragments at `conf >= 0.05`; a human reviewer is expected to
reclassify / delete many of them. This is a **sampling&labeling concern to fix
during review**, not a ground-truth fact.

## 5. Deduplicated object counts

Boxes that are byte-identical duplicates (same class + rounded box) are dropped
in `build_evaluation_set.py`, so the counts above are already deduplicated.

## 6. Conclusion & plan

- The current 150-frame set is heavily positive-biased. Do **not** treat it as a
  balanced accuracy test on its own.
- **Do not delete** the existing 150 frames. Instead **add independent negative /
  background frames** without using model detection as the selection criterion.
  Proposed: +5 frames per video at **interleaved timestamps** (half the existing
  1 s spacing) and running the LABELS UNVERIFIED so a human can mark them 0-object.
- Respect `--frames-per-video` overrides; keep total around 200-300.
- Do **not** auto-hit 1000 frames — quality of ground truth > quantity.

Details of the negative-frame plan:

- Keep the existing 150.
- Add ~5 additional frames per video at interleaved timestamps to broaden scene
  and background coverage (total ~200).
- These new frames get pre-labels ONLY if the model emits low-conf boxes, but the
  human review will decide the final 0-object mark.

## 7. Updated measurement after background-frame pass

After running `scripts/add_background_frames.py` (5 extra interleaved frames per
video, selected WITHOUT using model detections):

| Measure | Before | After |
|---|---:|---:|
| Total frames | 150 | 200 |
| Positive (has box) | 138 | 185 |
| Negative / background | 12 | 15 |
| Negative ratio | 8.0% | 7.5% |

Despite adding 50 interleaved frames, the negative ratio did not improve
because the QA source videos are **object-dense by design** (a subtitle/watermark /
logo is present throughout). Sparse clean-background content cannot be conjured
from these clips.

**Honest conclusion:** the evaluation set remains positive-heavy. True negative /
background coverage requires additional **real-world videos** that contain
clean-background interstitials — that is a data-collection follow-up, not
something the current QA corpus can supply. This is a documented limitation of the
current evaluation set.
