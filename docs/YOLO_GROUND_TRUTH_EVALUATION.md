# YOLO Ground-Truth Evaluation Specification & Workflow

Document Version: 1.0.0
Date: 2026-08-07
Target Model: `models/yolo/best.pt` (YOLO26n fine-tuned 4-class detector)
SHA-256: `4BC3F57A9E6A6A2A4AD91F1F7037A73A290D90A8673B6F01EAC63AAA515C953C`

---

## 1. Overview

This document specifies the quantitative evaluation protocol for the fine-tuned 4-class YOLO burn-in text detector in `media-flow`.

### Honesty Contract
No model predictions are ever treated as ground truth. Metrics (Precision, Recall, F1, localization IoU, confusion matrix) are computed **only** against verified ground-truth labels (`verified=true`). If pre-labels have not yet been reviewed by a human, the evaluation pipeline outputs `PENDING_HUMAN_REVIEW` and halts without fabricating performance numbers.

---

## 2. Dataset & Sampling Architecture

The evaluation set is managed under `evaluation/`:

```text
evaluation/
├── frames/                 # Extracted JPEG frame images
├── labels/                 # Canonical JSON labels ({frame_id, video_id, objects, verified})
├── manifests/
│   ├── manifest.csv        # Frame-level manifest (frame_id, video_id, frame_index, timestamp_seconds, width, height, verified)
│   └── exclusion_manifest.csv # Excluded videos/frames and leakage prevention record
├── review/
│   ├── previews/           # Pre-label visualization frames with boxes drawn
│   ├── review_index.html   # Browser inspection interface
│   └── README.md           # Human review instructions
└── README.md
```

### 2.1 Sampling Strategy & Temporal Leakage Prevention
1. **Video-Level Split**: Frames are sampled per video with a minimum temporal gap (default: 1.0 s / 15-30 frames). Frames from the same video are **never** split between training and test sets.
2. **Exclusion Manifest**: `evaluation/manifests/exclusion_manifest.csv` records all excluded videos (synthetic benchmark clips, legacy QA clips) and explicitly documents training provenance safety checks.

---

## 3. Metrics Definition

### 3.1 Bounding Box Matching Protocol
- **Matching Algorithm**: One-to-one greedy descending-IoU matching.
- **Class-Aware Matching** (used for Precision, Recall, F1): Only predictions matching the ground truth's `class_id` are eligible. Wrong-class overlaps produce 1 FN (for the GT class) and 1 FP (for the predicted class).
- **Class-Agnostic Matching** (used for the Confusion Matrix & Error Taxonomy): Overlaps are matched regardless of predicted class to capture confusion (`subtitle` $\leftrightarrow$ `other_burn_in_text`, `watermark` $\leftrightarrow$ `logo`).
- **IoU Thresholds**: Default evaluation runs at $\text{IoU} \ge 0.50$ and $\text{IoU} \ge 0.75$.

### 3.2 Metric Formulas
- **Precision**: $P = \frac{TP}{TP + FP}$
- **Recall**: $R = \frac{TP}{TP + FN}$
- **F1 Score**: $F_1 = \frac{2 \cdot P \cdot R}{P + R}$
- **Micro Average**: Recomputed $P, R, F_1$ across total $\sum TP, \sum FP, \sum FN$.
- **Macro Average**: Unweighted mean of per-class $P, R, F_1$.
- **Localization Stats**: Mean IoU, Median IoU, $P_{10}$ IoU, $P_{90}$ IoU of matched pairs.

---

## 4. Class Action Policy & Downstream Routing

The detector separates **detection** from **action policy**:

```yaml
class_policy:
  subtitle:
    detect: true
    safe_cover: true
  watermark:
    detect: true
    safe_cover: true
  logo:
    detect: true
    safe_cover: true
  other_burn_in_text:
    detect: true
    safe_cover: true
```

- `detect: false`: Suppresses box emission from the detector entirely.
- `safe_cover: false`: Detector emits the box, but downstream inpainting/covering modules treat it as read-only (not to be covered or removed).
- **CLI Overrides**:
  - `python scripts/test_video_detection.py --classes subtitle,other_burn_in_text` (restricts detection)
  - `python scripts/test_video_detection.py --cover-classes subtitle` (restricts covering action)

---

## 5. Performance Timing Instrumentation

To ensure honest timing metrics, `scripts/run_detection_benchmark.py` separates raw model execution from full pipeline overhead:

- `raw_model_fps`: $\frac{\text{Processed Frames}}{\text{Inference Time (\text{model.predict})}}$
- `pipeline_fps`: $\frac{\text{Processed Frames}}{\text{Total Wall Time (Decode + Detect + Render + Encode)}}$

### Timing Breakdown Metrics:
1. `decode_seconds`: OpenCV frame reading
2. `inference_seconds`: Torch FP16 CUDA execution
3. `postprocess_seconds`: Confidence filtering, clamping, duplicate NMS, expansion
4. `render_seconds`: Drawing bounding boxes / generating masks
5. `mux_seconds`: FFmpeg audio copying / AAC re-encoding

---

## 6. Reproduction Commands

### 1. Re-build evaluation dataset from videos
```powershell
.venv\Scripts\python.exe scripts\build_evaluation_set.py `
    --input-dir qa_inputs `
    --out-dir evaluation `
    --frames-per-video 15 `
    --device cuda
```

### 2. Validate evaluation label schema
```powershell
.venv\Scripts\python.exe scripts\validate_evaluation_labels.py `
    --manifest evaluation\manifests\manifest.csv `
    --labels evaluation\labels `
    --frames evaluation\frames
```

### 3. Run evaluation (requires verified=true in manifest)
```powershell
.venv\Scripts\python.exe scripts\evaluate_yolo_detection.py `
    --manifest evaluation\manifests\manifest.csv `
    --labels evaluation\labels `
    --frames evaluation\frames `
    --out-dir outputs\evaluation `
    --device cuda
```

### 4. Run model verification
```powershell
.venv\Scripts\python.exe scripts\verify_yolo_model.py --device cuda
```

### 5. Run full test suite
```powershell
.venv\Scripts\python.exe -m pytest tests/ -q
```
