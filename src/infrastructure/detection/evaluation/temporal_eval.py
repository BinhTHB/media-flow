"""Temporal continuity metrics for detection over time.

Frame-level metrics compare a per-frame "object present" signal (ground truth)
against the detector's per-frame presence. They are only meaningful when the
ground truth covers a contiguous run of frames (e.g. a subtitle event spanning
20 frames). Sampling isolated frames can only give a coarse view; document that
limitation where these metrics are used.

Metrics:

* ``coverage_ratio`` — fraction of GT-positive frames the detector also marked
  present (19/20 detection of a 20-frame event is 0.95, 10/20 is 0.50).
* ``max_consecutive_missed`` — longest run of consecutive GT-positive frames
  with no detection (miss streak).
* ``flicker_rate`` — how often the detector presence toggles while GT presence
  is steady (spurious on/off switching within a stable event).
"""

from __future__ import annotations

from typing import Dict, List, Sequence


def _bool_seq(values: Sequence[object]) -> List[bool]:
    return [bool(v) for v in values]


def compute_frame_coverage(
    gt_present: Sequence[bool],
    pred_present: Sequence[bool],
) -> Dict[str, float]:
    """Frame-level temporal metrics over aligned per-frame presence signals.

    ``gt_present`` and ``pred_present`` must be the same length, aligned by
    frame index. Returns dict with coverage_ratio, max_consecutive_missed,
    flicker_rate, n_gt_frames, n_detected.
    """
    gt = _bool_seq(gt_present)
    pred = _bool_seq(pred_present)
    if len(gt) != len(pred):
        raise ValueError("gt_present and pred_present must be aligned (same length)")
    n_gt = sum(gt)
    detected = sum(1 for g, p in zip(gt, pred) if g and p)
    coverage = detected / n_gt if n_gt else 0.0

    # longest consecutive missed run inside GT-positive stretches
    max_miss = 0
    run = 0
    for g, p in zip(gt, pred):
        if g and not p:
            run += 1
            max_miss = max(max_miss, run)
        else:
            run = 0

    # flicker: presence toggles while GT is steadily present (>=3 frames)
    toggles = 0
    steady_frames = 0
    i = 0
    n = len(gt)
    while i < n:
        if gt[i]:
            j = i
            while j < n and gt[j]:
                j += 1
            run_len = j - i
            if run_len >= 3:
                steady_frames += run_len
                for k in range(i + 1, j):
                    if pred[k] != pred[k - 1]:
                        toggles += 1
            i = j
        else:
            i += 1
    flicker = toggles / steady_frames if steady_frames else 0.0

    return {
        "coverage_ratio": round(coverage, 4),
        "max_consecutive_missed": float(max_miss),
        "flicker_rate": round(flicker, 4),
        "n_gt_frames": float(n_gt),
        "n_detected": float(detected),
    }


def event_coverage(
    gt_frame_indices: Sequence[int],
    pred_frame_indices: Sequence[int],
    total_frames: int,
) -> Dict[str, float]:
    """Coverage of one event (a set of GT-present frame indices).

    ``total_frames`` is the video frame count; ``pred_frame_indices`` is the set
    of frames the detector marked present anywhere in the video. Computes what
    fraction of the event's frames were detected and the longest internal miss
    streak.
    """
    gt_set = set(int(i) for i in gt_frame_indices)
    pred_set = set(int(i) for i in pred_frame_indices)
    if not gt_set:
        return {"coverage_ratio": 0.0, "max_consecutive_missed": 0.0,
                "n_gt_frames": 0.0, "n_detected": 0.0}
    gt_present = [i in gt_set for i in range(total_frames)]
    pred_present = [i in pred_set for i in range(total_frames)]
    return compute_frame_coverage(gt_present, pred_present)
