# Phase 9B Same-Source Baseline

## Scope

Phase 9B aligns provenance only. It does not change adaptive gates, static
burn, sidecar generation, watermark handling, TTS, or strict defaults.

## Source investigation

The V5 report at
`output/adaptive_render_v5/debug/adaptive_render_report.json` records
`sample.mp4` as its input. No file named `sample.mp4` exists in the project.
The V5 rendered artifact probes as 1440x2560, 30 fps, and 22.616009 seconds,
but that does not establish the missing input's identity.

The canonical source at
`test_videos/real_mini_benchmark/sample1.mp4` probes as 1440x2560, 30 fps,
and 22.616236 seconds. The existing current V6 video at
`output/subtitle_policy_v6/sample1_adaptive_source_replace/videos/output.mp4`
probes as 1080x1920, 30 fps, and 38.916644 seconds. Although its report
records the canonical `sample1.mp4` path, its media properties and 38
one-second analysis intervals do not match the canonical file.

**Same-source determination:** the legacy V5 and current V6 artifacts are not
proven same-source artifacts. Their prior direct comparison is invalid for
visual, candidate, gate, or safety conclusions.

## Phase 9B baseline

The existing adaptive renderer was rerun against the canonical source:

```powershell
& .venv\Scripts\python.exe -m scripts.render_adaptive_subtitles `
  --video test_videos\real_mini_benchmark\sample1.mp4 `
  --captions output\real_video_benchmark_results\sample1\transcripts\normalized.json `
  --region-report output\real_video_benchmark_results\sample1\debug\adaptive_subtitle_region_report.json `
  --interval-report output\real_video_benchmark_results\sample1\debug\interval_eligibility_report.json `
  --output-root output\phase9b_same_source_baseline `
  --coverage-mode adaptive_source_replace `
  --review-timestamps 5,10,15 `
  --ffmpeg-encoder cpu
```

The output at
`output/phase9b_same_source_baseline/videos/output.mp4` preserves the source
resolution, frame rate, and duration: 1440x2560, 30 fps, 22.616236 seconds.
The report contains three accepted replacements, but only `review_13` lies
within the canonical source duration. This confirms that the reused
38-interval analysis artifacts are not fully aligned with the canonical
source.

The render report also declares `play_res_x=1080` and `play_res_y=1920`,
while the actual input is 1440x2560. Therefore this artifact proves that the
video file passed to the renderer is the canonical source, but it is **not**
a valid same-source geometry/candidate baseline. Its region, interval, and
caption artifacts came from the stale 1080x1920/38.916644-second analysis.
It must not be used to tune gates.

The valid source/output visual comparison is
`output/phase9b_same_source_baseline/review_frames/contact_sheet_source_vs_baseline_5s_10s_15s.jpg`.
Structured evidence is in
`output/phase9b_same_source_baseline/comparison_report.json`.

## Verdicts

- Comparison validity, legacy V5 versus current V6: **INVALID**
- Same-source baseline: **FAIL**
- Canonical source selected for the future baseline:
  `test_videos/real_mini_benchmark/sample1.mp4`
- Current Phase 9B render: provenance-aligned video input, but stale geometry;
  retain as diagnostic evidence, not as the golden baseline
- Tuning readiness: **NOT READY**
- Production Ready: **NO**

Before threshold or gate tuning, regenerate OCR/localization, caption
timeline, region report, interval eligibility report, adaptive render, and
review artifacts from the canonical `sample1.mp4`. Only that complete
same-source chain can become the future baseline.
