# Backend Processing Architecture Specification

## Overview

This document specifies the **Policy-Driven Media Processing Architecture** of `media-flow`.
The system is built as a clean, extensible, class-aware pipeline that decouples object detection from business actions and rendering mechanisms.

```text
Input Video
    ↓
YOLO 4-class Detector (models/yolo/best.pt)
    ↓
Class-Aware Spatial-Temporal Tracker
    ↓
Action Policy Engine (ProcessingPolicyConfig)
    ↓
Action Plan / RenderingPlan (outputs/<job>/action_plan.json)
    ↓
Lazy Enrichment Service (OCR / STT / Translation / Asset Resolution)
    ↓
Action Video Renderer (Pixel Blur / Solid Cover / Image Replace / Text Replace)
    ↓
Audio Stream Muxing & Invariant Validation
    ↓
Final MP4 Video
```

---

## 1. Class-Aware Canonical Detections & Tracking

The detector (`models/yolo/best.pt`) emits 4 canonical classes:
- `0`: `subtitle`
- `1`: `watermark`
- `2`: `logo`
- `3`: `other_burn_in_text`

Unlike legacy pipelines that collapsed detections into generic `"text"`, the policy engine preserves `class_id` and `class_name` through every stage of processing. The `ClassAwareTracker` tracks spatial-temporal continuity across frame boundaries for each class independently without applying restrictive coordinate gates.

---

## 2. Processing Actions & Behavior Matrix

The `ActionPolicyEngine` evaluates every tracked object against configured class policies.

### Supported Processing Actions

| Action | Execution Behavior | Expensive Dependencies Required |
|---|---|---|
| `blur` | Real Gaussian pixel blur over bounding box regions per frame | None (100% Offline) |
| `solid_cover` | Solid color rectangle fill (`solid_color_hex`) over regions | None (100% Offline) |
| `replace_image` | Alpha-composited PNG/JPEG asset overlay fitted over bbox | Image Asset File |
| `replace_text` | OCR text extraction, translation, and Unicode text rendering | Gemini OCR & Gemini Translation |
| `preserve` | Keeps original video pixels completely untouched | None (100% Offline) |

### Class Action Matrix

| YOLO Class | Default Action (`blur-all`) | `translate-subtitle` Preset | Custom Action Options |
|---|---|---|---|
| `subtitle` | `blur` | `replace_text` (target=`vi`) | `blur`, `solid_cover`, `replace_text`, `replace_image`, `preserve` |
| `watermark` | `blur` | `preserve` | `blur`, `solid_cover`, `replace_image`, `replace_text`, `preserve` |
| `logo` | `blur` | `preserve` | `blur`, `solid_cover`, `replace_image`, `preserve` |
| `other_burn_in_text` | `blur` | `preserve` | `blur`, `solid_cover`, `replace_text`, `replace_image`, `preserve` |

---

## 3. Presets and Configuration

### Pre-built Presets
- `blur-all` (Default): All 4 classes mapped to `blur`. Runs fully offline without API keys.
- `translate-subtitle`: `subtitle` mapped to `replace_text`, all other classes set to `preserve`.
- `preserve-all`: All classes set to `preserve`.

### Custom Policy YAML Example (`config/custom_policy.yaml`)

```yaml
classes:
  subtitle:
    enabled: true
    action: replace_text
    confidence_threshold: 0.15
    text_source: ocr
    target_language: vi
  watermark:
    enabled: true
    action: blur
    blur_strength: 30.0
    padding_px: 6
  logo:
    enabled: true
    action: replace_image
    image_asset_path: assets/brand_logo.png
    opacity: 0.9
  other_burn_in_text:
    enabled: true
    action: solid_cover
    solid_color_hex: "#000000"
```

---

## 4. Application Boundary & Frontend Readiness

The application exposes a single unified entry service:

```python
from src.application.dto.processing_request import ProcessingRequest
from src.infrastructure.services.media_processing_service import MediaProcessingService

request = ProcessingRequest(
    input_video="input.mp4",
    output_video="output.mp4",
    preset="blur-all",  # or custom policy_config / policy_file
    device="auto",
)

service = MediaProcessingService()
result = service.process_video(request)
```

Every execution produces a structured `action_plan.json` under the output directory containing track-by-track window actions for inspection and Web UI rendering.
