# Gemini Flash-Lite Multi-Stage OCR, Phase G.3

## Status

Production Ready: **NO**

## Root cause

The previous full-frame OCR request combined region detection, text classification,
transcription, reading order, and geometry. A wrong region could therefore
contaminate transcription, grouping, translation, and cover geometry.

## Multi-stage architecture

All OCR stages use the shared `GeminiRuntime` and
`gemini-3.1-flash-lite`. Local processing is limited to image crops, resize,
geometry conversion, tracking, temporal smoothing, caching, and rendering.

1. Stage A detects text regions without transcription or classification.
2. Deterministic tracking links compatible observations into spatial tracks.
3. Stage B classifies tracks before any text is accepted.
4. Stage C transcribes crops for subtitle tracks only.
5. Stage D verifies unresolved disagreement, at most once per track.

## Focused requests

### Request A, detection

Detection requests return bounded regions only. Invalid coordinates, unknown
frames, and zero-area boxes are rejected.

### Request B, classification

Classification decides whether a region is output text or a rejected overlay.
Rejected and review-required regions remain in debug artifacts.

### Request C, crop recognition

Recognition receives an isolated crop. It returns only verbatim text and a
readability state. Empty or ambiguous text is not emitted as an OCR segment.

### Request D, verification

Verification is optional and bounded. A conflicting or unreadable result is
review-required, not rendered.

## Cache and request budget

Each Gemini request is issued through the existing runtime, which provides the
shared scheduler, REST client, response cache, model validation, and usage
recorder. Stage-specific source kinds and prompt/schema versions isolate cached
responses. Live runs must set an explicit request budget before execution.

## Tracking and intervals

Tracks use deterministic IoU-based continuity and retain track provenance.
Only evidence-gated, readable subtitle observations become OCR `Segment`
instances. Rejected or review-required observations are excluded from
translation and replacement rendering.

## Translation and dynamic replacement

The existing translation provider translates only returned OCR intervals. In
`replace_source_subtitle` mode, the orchestrator emits display subtitles from
OCR segments only, preserving Layer 0 cover and Layer 1 translated text timing.
ASR-only display events are excluded.

## Cover guards

The cover geometry originates from the accepted OCR segment bbox. Invalid OCR
geometry cannot reach rendering. Oversized-cover and visual safety acceptance
remain live-review gates.

## No-local-OCR proof

`GeminiMultiStageOCRProvider` uses Gemini requests for detection,
classification, transcription, and verification. It imports no PaddleOCR,
Tesseract, EasyOCR, OpenCV, or local text detector. Local image operations use
Pillow only for permitted crop and resize work.

## Evidence artifacts

For a multistage job, the orchestrator writes:

- `debug/g3_region_windows.json`
- `debug/g3_region_tracks.json`
- `debug/g3_region_classifications.json`
- `debug/g3_crop_ocr_results.json`
- `debug/g3_verification_results.json`
- `debug/g3_ocr_intervals.json`
- `debug/g3_rejected_tracks.json`
- `debug/g3_review_required.json`

## Live sample status

Sample2, sample4, and sample5 live G.3 validation remains pending. The required
review bundle must include track review images, before/after contact sheets,
request counts, interval coverage, replacement events, and sample5 dynamic
person-overlap safety evidence.

## Known limitations

Offline tests validate the provider contract, tracking, artifact persistence,
and legacy compatibility. They cannot establish OCR linguistic accuracy,
subtitle coverage, or person-overlap safety on real media. These require the
pending Gemini live runs and human visual review.
