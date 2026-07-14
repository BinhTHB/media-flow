# Phase 9 V5 Replay Audit

## Scope

This is an audit-only replay of `adaptive_render_v5` against the externally
modified V6 gate report. It does not change renderer, policy, static burn,
sidecar, watermark, TTS, or default behavior.

## Evidence

- V5 report: `output/adaptive_render_v5/debug/adaptive_render_report.json`
- Current V6 report: `output/subtitle_policy_v6/sample1_adaptive_source_replace/debug/adaptive_render_report.json`
- Current V6 gate: `output/subtitle_policy_v6/sample1_adaptive_source_replace/debug/gate_audit_report.json`

V5 has eight `expanded_line_bbox` replacements from 0.0 to 19.3 seconds.
Each recorded coordinate mapping `PASS`, no clipping, mask area from 0.046106
to 0.06191, and `source_leakage_detected: not_evaluated`.

V6 has 38 one-second review intervals and three adaptive replacements,
`review_13`, `review_31`, and `review_36`. Only `review_13` overlaps the V5
time range. The strict video gate is
`BLOCK_SOURCE_REPLACE/no_high_confidence_source_replace_intervals`; the
adaptive gate is `AUTO_SOURCE_REPLACE`.

The recorded video paths differ: V5 reports `sample.mp4`, while V6 reports
`test_videos/real_mini_benchmark/sample1.mp4`. The generated contact sheet
confirms different visual content at 5, 10, and 15 seconds. Timestamp mapping
is complete, but this provenance mismatch prevents a claim of candidate or
safety equivalence.

## Timestamp Replay

| V5 caption | V5 time | Current overlapping intervals | Primary cause |
| --- | ---: | --- | --- |
| `speech_0_0` | 0.000-3.337 | `review_0`-`review_3` | `no_reliable_source_subtitle` |
| `speech_0_1` | 3.337-5.760 | `review_3`-`review_5` | `insufficient_track_evidence` |
| `speech_2` | 5.760-8.400 | `review_5`-`review_8` | `insufficient_track_evidence` |
| `ocr_7_1` | 8.400-9.900 | `review_8`-`review_9` | `insufficient_track_evidence` |
| `speech_3_0` | 9.900-12.441 | `review_9`-`review_12` | `insufficient_track_evidence` |
| `speech_3_1` | 12.441-13.860 | `review_12`, `review_13` | `insufficient_track_evidence` |
| `speech_5_0` | 13.860-17.442 | `review_13`-`review_17` | `no_reliable_source_subtitle` |
| `speech_5_1` | 17.442-19.300 | `review_17`-`review_19` | `no_reliable_source_subtitle` |

The full per-caption evidence is in
`output/phase9_v5_replay_audit/gate_audit_report.json`. It records the
candidate wiring, strict/adaptive video gate, track result, target-text
presence, safety evidence availability, and override readiness for every
overlapping interval.

## Cause Distribution

Primary cause by V5 caption:

- `insufficient_track_evidence`: 5
- `no_reliable_source_subtitle`: 3

`missing_target_text` appears in the overlap evidence for `speech_5_0` and
`speech_5_1`, but it is not the primary cause for either caption.

## Verdicts

- V5 compatibility parity: **NO**
- Standalone replay audit: **COMPLETE**
- Production Ready: **NO**

The V5 artifact supplies local geometry evidence, but the source video
provenance differs and neither the V5 report nor the overlapping non-rendered
V6 intervals evaluate source leakage, person overlap, watermark overlap, or
scene damage. Candidate-level replay against the same video is required
before an override can be approved.
