# Static Source Cover MVP Baseline Specification

## 1. Goal of the MVP
Decouple and resolve the core translation replacement issue by limiting the scope to video subtitles with stable locations (static subtitles). The system detects the exact localized lane/region occupied by the source subtitle, covers it with a matching solid box/band, and burns the target/translated subtitle over the covered region. This guarantees a clean replacement without floating/adaptive overlay artifacts.

## 2. Out-of-Scope
- **Dynamic/floating subtitle replacement**: Tracking and covering moving/bouncing subtitles.
- **Bubble/caption overlay removal**: Complex frame restoration and inpainting.
- **Watermark removal / Person face masking**: Heavy visual editing steps.
- **Model fine-tuning**: Custom layout segmentation or OCR training.

## 3. Main Modes

### A. Static Source Cover (`static_source_cover`)
- **Action**: Detects a stable subtitle lane via location, size, and style consistency analysis. Covers the original subtitle with an expanded semi-transparent black box and renders target text over the same region.
- **Validation placeholder**: Uses `DỊCH_n` target text placeholders when translations are unavailable.

### B. Static Subtitle Burn (`static_subtitle_burn`)
- **Action**: Renders static subtitles in a readable bottom-center position without covering the original text, as a fallback mode when no stable region can be cleanly masked.

### C. Sidecar Subtitles (`sidecar`)
- **Action**: Always generates ASS (`output.ass`) and SRT (`output.srt`) subtitle tracks alongside the video tracks.

---

## 4. Sample Verification Baseline Results

| Sample | Stable Region Found | Rendered Count | Skipped Count | Duplicate Risk | Visual Verdict |
|---|---|---:|---:|---|---|
| **sample2** | True | 22 | 0 | Low | **PASS** |
| **sample4** | True | 25 | 0 | Low | **PASS** |
| **sample5** | True | 24 | 1 | Low | **PASS** |

- *Note*: `sample5` has 1 interval (`interval_6`) correctly skipped with `true_person_overlap_risk` because an athlete's body fully obstructs the subtitle region.

---

## 5. Known Limitations
- Stable region detection expects at least two frames of temporal coverage. Extremely short static subtitles may fall back to `static_subtitle_burn`.
- Large foreground motion (e.g. runners passing over subtitles) triggers safety skips, leaving short segments uncovered.

## 6. Acceptance Criteria
- Stable regions must be localized correctly (no fixed-bottom hardcoding).
- Overlaps by central people or faces must be skipped (`true_person_overlap_risk`), while minor edge overlaps can be safely overridden.
- Output sidecar ASS/SRT files must be written for all samples.

## 7. Artifact Paths

Baseline outputs are organized under `output/static_source_cover_mvp_v2/<sample>/`:
- Main output: `videos/static_source_cover.mp4`
- Fallback output: `videos/static_subtitle_burn.mp4`
- Sidecars: `subtitles/output.ass`, `output.srt`
- Metadata: `debug/static_source_cover_report.json`, `debug/artifact_provenance_report.json`

---

## 8. Production Readiness
**Production Ready: NO**
This is an MVP freeze baseline showing localized cover capabilities. Target translation APIs and automated context formatting are needed for full production readiness.