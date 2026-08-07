# Backend Processing Architecture & API Contract Specification

## Overview

This specification defines the **Policy-Driven Media Processing Architecture** and **Frontend-Ready Backend API Contract** for `media-flow`.

The backend is completely decoupled from UI logic. A frontend or REST API wrapper invokes the backend via strongly-typed contract objects (`ProcessingRequest` $\rightarrow$ `ProcessingResult`) without needing any knowledge of YOLO, Gemini, OpenCV, or FFmpeg internals.

```text
Input Video
    ↓
YOLO 4-class Detector (models/yolo/best.pt)
    ↓
Class-Aware Spatial-Temporal Tracker (Preserves class_id & class_name: 0=subtitle, 1=watermark, 2=logo, 3=other_burn_in_text)
    ↓
Action Policy Engine (ProcessingPolicyConfig)
    ↓
Job Workspace Isolation (output/jobs/<job_id>/ containing job.json, detections.json, tracks.json, action_plan.json)
    ↓
Lazy Enrichment Service (OCR / STT / Gemini Translate / Asset Resolution - ONLY when requested)
    ↓
Action Video Renderer (Real Gaussian Pixel Blur / Solid Cover / Image Replace / Text Replace)
    ↓
Audio Stream Muxing & Invariant Validation
    ↓
Final Output MP4 Video
```

---

## 1. Backend API Contract (`ProcessingRequest` & `ProcessingResult`)

### `ProcessingRequest`
Exposes all frontend-tunable business parameters with sensible defaults:

```python
@dataclass
class ProcessingRequest:
    input_video: str
    output_video: str
    job_id: Optional[str] = None
    device: str = "auto"                     # auto | cuda | cpu
    preset: Optional[str] = None             # blur-all | translate-subtitle | preserve-all
    policy_config: Optional[ProcessingPolicyConfig] = None
    policy_file: Optional[str] = None
    speech_provider: str = "faster-whisper"
    translation_provider: str = "gemini"
    ocr_provider: str = "hybrid_gemini"
    target_language: str = "vi"
    overwrite: bool = True
```

### `ProcessingResult`
Clean, JSON-serializable output summary:

```python
@dataclass
class ProcessingResult:
    job_id: str
    status: str                              # SUCCESS | FAILED | BLOCKED
    input_video_path: str
    output_video_path: str
    duration_sec: float
    fps: float
    resolution: str
    detections_summary: Dict[str, int]
    tracks_summary: Dict[str, int]
    actions_summary: Dict[str, int]
    action_plan_path: str
    workspace_dir: str
    timings: Dict[str, float]
    warnings: List[str]
    error: Optional[str] = None
```

---

## 2. Job Workspace Isolation

Every execution runs in an isolated workspace under `output/jobs/<job_id>/`. Multiple jobs running concurrently or sequentially will never overwrite each other's metadata artifacts:

```text
output/jobs/<job_id>/
├── job.json            # ProcessingRequest payload & job metadata
├── detections.json     # Canonical detections list with class_id/class_name
├── tracks.json         # Spatial-temporal tracks list
├── action_plan.json    # Full machine-readable rendering plan
└── logs/               # Intermediate processing logs
```

The final processed video file is written directly to the exact path specified by `output_video`.

---

## 3. Disambiguation of Action Statuses (`blur`, `preserve`, `ignored`)

- **`active` (`blur` / `solid_cover` / `replace_text` / `replace_image`)**: The track met all policy criteria and an explicit action will be rendered on pixels.
- **`active` (`preserve`)**: The track met confidence thresholds, but the user policy explicitly specifies `action = "preserve"` (intentionally leave pixels untouched).
- **`ignored`**: The track did NOT meet the class confidence threshold (`mean_confidence < confidence_threshold`) or the class was disabled (`enabled = False`). Reported as `status = "ignored"` with `reason = "below_confidence_threshold"` or `"class_disabled"`.

---

## 4. Text Replacement Sources (`ocr`, `stt`, `hybrid`)

When `action = replace_text`:
- **`text_source = "ocr"`**: Runs Gemini OCR on YOLO subtitle crops + Gemini Translation. STT (Whisper) is NOT loaded or executed.
- **`text_source = "stt"`**: Extracts audio and runs Faster-Whisper STT. OCR is NOT executed. Matches speech segments to interval timestamps.
- **`text_source = "hybrid"`**: Runs Whisper STT for audio speech AND Gemini OCR for on-screen burn-in text. Matches intervals: if OCR text is present, on-screen subtitle replacement uses OCR text while speech narration uses STT text.

---

## 5. Supported Processing Actions & Tunable Parameters

| Action | Render Mechanism | Customizable Parameters |
|---|---|---|
| `blur` | Real Gaussian Pixel Blur (`cv2.GaussianBlur`) | `blur_strength` (default 25.0), `padding_px` (default 4) |
| `solid_cover` | Solid Rectangle Fill | `solid_color_hex` (default "#000000"), `padding_px` |
| `replace_image` | Alpha PNG/JPEG Overlay | `image_asset_path`, `fit` ("contain"/"cover"/"stretch"), `scale` (1.0), `opacity` (1.0), `z_order` |
| `replace_text` | Unicode Text Overlay | `text_source` ("ocr"/"stt"/"hybrid"), `target_language`, `font`, `font_size`, `stroke_width`, `placement`, `background_action` |
| `preserve` | Untouched Original Pixels | None |

---

## 6. Pre-built Presets

- **`blur-all`** (Default): All 4 YOLO classes (`subtitle`, `watermark`, `logo`, `other_burn_in_text`) mapped to `blur`. Runs 100% offline without requiring `GEMINI_API_KEY`.
- **`translate-subtitle`**: `subtitle` mapped to `replace_text` (target=`vi`), all other classes set to `preserve`.
- **`preserve-all`**: All classes mapped to `preserve`.
