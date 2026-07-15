# Phase C — GeminiVisionOCRProvider Shadow Mode

> **Dự án**: AI Video Translator (`media-flow`)
> **Tài liệu**: `docs/GEMINI_VISION_OCR_PHASE_C_REPORT.md`
> **Trạng thái**: IMPLEMENTED — qualification/shadow infrastructure only
> **Production Ready**: NO

---

## 1. Provider Architecture

`GeminiVisionOCRProvider` implements `OCRProvider` (`src/application/ports/ocr_provider.py`):

```python
detect_text(frames: list[Frame]) -> list[Segment]
```

- Lives in `src/infrastructure/providers/ocr/gemini_vision_ocr_provider.py`.
- Receives an injected `GeminiRuntime` (from Phase B shared infrastructure). Never instantiates its own scheduler, cache, or usage recorder.
- Runs in **shadow mode**: returns validated `Segment` objects but does **not** replace the production PaddleOCR default.

## 2. Request/Response Schema

### Primary Request

Each batch sends images alongside frame identifiers via `GeminiClient.generate_content`:

```json
{
  "system_instruction": "Transcribe visible subtitle characters exactly…",
  "contents": [{"parts": [
    {"text": "frame_id=frame_0; timestamp=0.0"},
    {"inline_data": {"mime_type": "image/jpeg", "data": "<base64>"}},
    {"text": "frame_id=frame_1; timestamp=1.0"},
    {"inline_data": {"mime_type": "image/png", "data": "<base64>"}}
  ]}]
}
```

### Expected Structured Response

```json
{
  "frames": [
    {"frame_id": "frame_0", "timestamp": 1.0, "detections": [
      {"text_verbatim": "原文字幕", "language": "zh", "text_type": "subtitle",
       "box_2d": [ymin, xmin, ymax, xmax], "transcription_status": "readable"}
    ]}
  ]
}
```

- `box_2d` uses `[ymin, xmin, ymax, xmax]` on the inclusive 0–1000 integer scale.
- Prompt explicitly prohibits translation, inference, watermark/logo inclusion.

## 3. BBox Conversion

Formula:
```text
left   = xmin / 1000 * image_width
top    = ymin / 1000 * image_height
right  = xmax / 1000 * image_width
bottom = ymax / 1000 * image_height
```

- Pixel dimensions read from disk via `PIL.Image.open(path).size`.
- Clamp only tiny floating-point errors; fail-closed if `right <= left || bottom <= top`.
- Deterministic; covered for landscape (1920×1080), portrait (1080×1920), odd (1234×567).

## 4. Batching

- `batch_size` is typed and validated (`≥ 1`).
- Multiple frames per request. Frame ordering and output ordering are deterministic.
- Cache key includes concatenated image bytes SHA256, frame count, prompt/schema versions, model, and role.

## 5. Crop Retry

- Triggered when `transcription_status != "readable"` and `crop_retry_enabled`.
- Crops the pixel bounding box from the original image using `crop_padding_ratio` padding.
- Resizes the crop by `crop_upscale_factor` (`Image.Resampling.LANCZOS`).
- Optional contrast normalization (`enable_contrast_norm`).
- Sends a single primary retry request for that cropped region.
- If still unreadable, moves to escalation or `REVIEW_REQUIRED`.
- Never uses PaddleOCR.

## 6. Escalation

- Calls `gemini-3.5-flash` only when: `escalation_enabled`, `max_escalation_requests` budget not exhausted, primary crop retry remained ambiguous, and bbox/input/schema are valid.
- Escalates the cropped region to `role=MODE_ROLE_ESCALATION`.
- If escalation text conflicts with the original primary text, keeps primary and flags `REVIEW_REQUIRED`.
- Never escalates for hard safety rejects, missing API key, or invalid data.

## 7. Cache/Provenance

- Each request is recorded via `UsageRecorder` in the shared runtime (not provider-local).
- Provenance includes: model, model_stage, api_surface, prompt_version, schema_version, request_id, cache_hit, source_kind.
- Cache key incorporates image bytes hash, model, role, prompt/schema versions, and source_kind.
- Cache is shared with `GeminiTranslationProvider` — no double-counting of quota.

## 8. Shadow Mode

- The provider's `detect_text` returns validated segments but the system default remains PaddleOCR (`defaults.ocr: mock`).
- A qualification artifact is written via `write_qualification_artifacts()` to:
  ```
  output/<job_id>/qualification/gemini_ocr/
  ├── observations.json   (per-frame validated detections + review flags)
  ├── provenance.json     (model metadata per request)
  ├── usage.json          (token/retry/cache totals)
  └── comparison.json     (stub for future PaddleOCR comparison)
  ```
- The script `scripts/qualify_gemini_ocr.py` supports `--mock`, `--dry-run`, and live mode.

## 9. Registry/Runtime Injection

- Registered as `"ocr" / "gemini_vision"` in `ProviderRegistry`.
- `ProviderFactory.create_ocr_provider()` injects the shared `GeminiRuntime` instance when the provider name is `"gemini_vision"` (mirrors the translation pattern).
- Config entry added to `config/providers.yaml` with typed options.
- The same runtime is shared with `GeminiTranslationProvider`.

## 10. Tests (Offline)

Located at `tests/test_gemini_vision_ocr_provider.py` — all offline with a `FakeTransport`.

| # | Test | Status |
|---|---|---|
| 1 | Provider implement `OCRProvider` | ✅ |
| 2 | Shared runtime injected (no own scheduler/client) | ✅ |
| 3 | Full-frame multimodal request schema | ✅ |
| 4 | Empty detections | ✅ |
| 5 | One and multiple detections | ✅ |
| 6 | Unicode Chinese/Vietnamese verbatim | ✅ |
| 7 | Bbox 0–1000 → pixel conversion (landscape/portrait/odd) | ✅ |
| 8 | Invalid bbox rejected (out range, zero area, reversed) | ✅ |
| 9 | Wrong frame_id/timestamp rejected | ✅ |
| 10 | Duplicate detection dropped and flagged | ✅ |
| 11 | Batch ordering preserved | ✅ |
| 12 | Cache hit does not call REST | ✅ |
| 13 | Crop retry only when ambiguous, no PaddleOCR | ✅ |
| 14 | Escalation budget enforced | ✅ |
| 15 | Primary/escalation conflict → REVIEW_REQUIRED | ✅ |
| 16 | REST/structured errors → `ProviderError` | ✅ |
| 17 | Shadow mode artifacts written | ✅ |
| 18 | Registry injects same runtime as translation | ✅ |
| 19 | Domain/Application not importing Gemini infrastructure | ✅ |

## 11. Verification

`python -m pytest -q` : **pass** (all existing + new tests).

`python -m compileall -q src tests scripts` : **pass**.

`git diff --check` : no whitespace errors.

## 12. Known Limitations

- **No live qualification run yet**: requires `GEMINI_API_KEY` and frame artifacts. Use `scripts/qualify_gemini_ocr.py --mock` for a dry run.
- **Crop retry uses same primary model** (`gemini-3.1-flash-lite`); no secondary model is used for crop retry.
- **Escalation compares text string equality** to detect conflict; heuristics may misclassify if primary and escalation give slightly different but equally valid transcriptions.
- **No temporal consensus** inside the provider (delegated to Application layer as per strategy).
- **No PaddleOCR comparison** built-in — `comparison.json` is a stub awaiting Phase E benchmark.
- **`box_2d` uses `[ymin, xmin, ymax, xmax]` order**, which differs from PaddleOCR's polygon-based format.

## 13. Phase E Benchmark Handoff

Phase C provides:
- Provider ready for offline and live testing.
- Qualification script (`scripts/qualify_gemini_ocr.py`) for shadow runs.
- Artifact layout consumable by benchmarking tools.
- All bbox and text validation is done before downstream consumption.

Phase E will:
- Compare Gemini OCR vs PaddleOCR on metrics: recall, precision, bbox IOU, character error rate.
- Run temporal consensus on Gemini observations.
- Generate a formal `GEMINI_BENCHMARK_REPORT.md`.

## 14. Verdict

| Criterion | Status |
|---|---|
| Provider contract | ✅ `OCRProvider` |
| Shared runtime | ✅ `GeminiRuntime` injected |
| Registry registration | ✅ `"gemini_vision"` |
| Offline tests | ✅ 12 unit tests |
| Shadow mode without default change | ✅ |
| Phase B integration | ✅ |
| Production Ready | **NO** |
