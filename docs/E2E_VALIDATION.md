# End-to-End Validation

E2E validation checks product behavior and architecture independence. It is not a YOLO model-quality benchmark.

## 1. General media invariants

For every produced video verify at minimum:

- output exists and decodes;
- duration is within the accepted tolerance of the expected timeline;
- expected resolution is preserved unless explicitly changed;
- expected FPS/timebase behavior is preserved;
- expected audio stream exists;
- audio/video start offset is acceptable;
- no unexpected blank/gray frames;
- job/result metadata reports the actual output path and status.

## 2. Scenario A — Visual blur-only

Purpose: prove Visual can run without Gemini/ASR/TTS.

Configuration concept:

```text
audio.enabled = false
visual.enabled = true
subtitle = blur-only or disabled
logo = blur
watermark = blur
other_burn_in_text = blur
```

Acceptance:

- YOLO and tracking run;
- enabled classes use shared BlurAction;
- original audio is preserved;
- no OCR/Translation/ASR/TTS provider initializes;
- must not require `GEMINI_API_KEY` when no Gemini-dependent visual operation is enabled.

## 3. Scenario B — Visual subtitle translate/replace

Purpose: validate the canonical subtitle path.

```text
Audio disabled
Visual enabled
Subtitle translate_replace
Logo blur
Watermark blur
Other blur
```

Acceptance:

- subtitle class routes only to SubtitleProcessor;
- subtitle track/event timing is preserved;
- OCR receives subtitle crop/evidence generated from YOLO geometry;
- Translation receives OCR text, not pixels;
- old subtitle pixels are blurred/covered by the configured background action;
- translated Vietnamese text is drawn at the source subtitle position/anchor;
- moving subtitles follow their source geometry;
- text uses adaptive wrapping/font sizing rather than blindly overflowing the original small bbox;
- logo/watermark/other do not enter the subtitle OCR path.

## 4. Required same-position text-change test

Include a fixture/video where two captions appear consecutively at the same or nearly the same bbox.

Expected:

```text
Track T may remain spatially continuous
but SubtitleEventSegmenter emits Event A and Event B
and each event receives its own OCR/translation text/timing.
```

A single OCR result reused across both captions is a failure.

## 5. Scenario C — Audio-only

Run when a real TTS production implementation is available.

```text
visual.enabled = false
audio.enabled = true
ASR → Translation → TTS → AudioComposer
```

Acceptance:

- YOLO is not initialized;
- visual frame processing does not run;
- ASR timestamps become speech segment timeline input;
- Translation is shared service/provider contract;
- TTS clips are composed against segment timing;
- final media duration remains valid;
- failures are reported honestly if TTS/provider support is incomplete.

Do not mock this scenario and report it as real E2E success.

## 6. Scenario D — Full Audio + Visual

Run after A–C are independently green.

```text
AudioPipeline enabled
VisualPipeline enabled
→ final mux
```

Acceptance:

- both feature results are combined without one pipeline controlling the internals of the other;
- final rendered visual stream uses the final selected audio stream;
- disabling either pipeline produces the corresponding independent behavior without code changes.

## 7. Module independence tests

The automated test suite should assert at least:

- `AudioPipeline` does not import/init YOLO;
- `VisualPipeline` does not import/init ASR/TTS;
- Subtitle disabled means no subtitle OCR/translation;
- blur-only means no Gemini runtime is required;
- `replace_image` does not initialize OCR/Translation;
- subtitle OCR result retains originating `event_id`, `track_id`, timing and geometry;
- Translation does not mutate bbox/timing identity;
- all non-subtitle visual classes can reuse the same BlurAction implementation;
- production packages do not import benchmark/evaluation packages.

## 8. Manual visual review

For subtitle replacement inspect representative frames for:

- source subtitle removed/blurred sufficiently;
- translated text visually corresponds to the correct source event;
- translated text position is anchored to the original subtitle region;
- no forced global bottom-center fallback;
- no unreasonable text overflow;
- no class-crossing actions such as logo text entering subtitle translation.

For blur classes inspect temporal stability and obvious false region damage.

## 9. Before/after refactor comparison

Before destructive cleanup, capture at least one known current E2E baseline. After migration run the same input/config and compare structural invariants and intended feature behavior.

The goal is regression detection, not model Precision/Recall/mAP evaluation.

## 10. Generated outputs

Write manual-review artifacts under a gitignored path such as:

```text
outputs/manual_review_feature_refactor/
```

Do not commit generated MP4/WAV/job-cache artifacts.
