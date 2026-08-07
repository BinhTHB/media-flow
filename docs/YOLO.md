# YOLO 4-Class Burn-In Detection

## 1. Production model baseline

Current accepted model baseline:

```text
models/yolo/best.pt
```

Model family: fine-tuned YOLO26n object detector.

Expected class mapping:

| class_id | class_name |
|---:|---|
| 0 | `subtitle` |
| 1 | `watermark` |
| 2 | `logo` |
| 3 | `other_burn_in_text` |

The detector must fail fast if the loaded model does not expose the expected four-class mapping.

## 2. YOLO responsibility

YOLO owns only visual detection facts:

```text
class_id
class_name
confidence
bbox
frame index
timestamp
```

YOLO does not:

- OCR text;
- translate text;
- decide blur/replace actions;
- call Gemini;
- decide TTS/audio behavior.

Do not collapse all classes into a generic `text` class downstream.

## 3. Canonical detection contract

A serialized detection should contain equivalent information to:

```json
{
  "class_id": 0,
  "class_name": "subtitle",
  "confidence": 0.8164,
  "bbox_xyxy": [291.7, 929.5, 423.7, 1060.8],
  "bbox_normalized": [0.405, 0.726, 0.588, 0.829],
  "frame_index": 3,
  "timestamp_seconds": 0.1,
  "frame_width": 720,
  "frame_height": 1280,
  "track_id": null
}
```

No Ultralytics result object may leak beyond the detector adapter.

## 4. Detection configuration

The existing model integration historically used per-class thresholds such as:

```yaml
confidence:
  subtitle: 0.15
  watermark: 0.20
  logo: 0.20
  other_burn_in_text: 0.20
```

These are product/config defaults, not immutable model constants. Runtime configuration may override them.

Do not reintroduce one universal threshold for every class unless later evidence supports it.

## 5. Device

Recommended device contract:

```text
auto | cuda | cpu
```

`auto` may choose CUDA when available. A user-requested explicit CUDA mode should fail clearly if CUDA is unavailable rather than silently changing execution semantics.

FP16 is a runtime optimization for supported CUDA execution and must not be forced on CPU.

## 6. Class-aware tracking

Tracking is separate from detection.

Requirements:

- match only same-class detections by default;
- retain `class_id` and `class_name` throughout the track;
- tolerate short configurable detection gaps;
- optionally smooth bbox trajectories;
- expire tracks rather than creating long ghost regions.

Tracking must not contain a universal bottom-screen rule for subtitles.

## 7. Subtitle tracking is not subtitle event segmentation

A spatial track can persist while the displayed text changes.

Example:

```text
00:01–00:03  caption A at bbox X
00:03–00:05  caption B at the same bbox X
```

A geometry-only tracker may produce one track. `SubtitleEventSegmenter` must split that track into distinct subtitle events before OCR/translation/rendering.

This separation is mandatory in V2.

## 8. Class routing

After tracking:

```text
subtitle            → SubtitleProcessor
watermark           → WatermarkProcessor
logo                → LogoProcessor
other_burn_in_text  → OtherTextProcessor
```

Default product actions:

```text
subtitle            → translate_replace
watermark           → blur
logo                → blur
other_burn_in_text  → blur
```

The detector does not own these defaults.

## 9. BBox safety

Before any destructive pixel action:

- validate non-empty bbox;
- clamp to frame bounds;
- reject invalid/inverted geometry;
- apply configured minimum-size rules where necessary;
- keep timestamps/frame mapping deterministic.

These geometry checks are lightweight safety rules and should remain even though older complex safety-policy layers are removed from the production path.

## 10. Model verification

Keep or provide a small verification command/script that checks:

- weights exist;
- model loads;
- task is detection;
- expected class mapping;
- configured device is usable;
- a dry-run can execute.

Model evaluation/ground-truth benchmarking is not part of the normal product execution path.

## 11. Model replacement

Replacing the detector model should require only:

1. place new weights in the model directory;
2. update model path/config;
3. keep or deliberately version the class contract;
4. run model-load verification and E2E regression.

Do not require changes to Audio, OCR, Translation, TTS, or class processors when the detection contract is unchanged.
