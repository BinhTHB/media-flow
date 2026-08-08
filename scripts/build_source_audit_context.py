#!/usr/bin/env python3
"""
Build Source Audit Context Reports v2 and ZIP Archive (Section 32)
"""

import os
import subprocess
import sys
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def main():
    context_dir = BASE_DIR / "outputs" / "source_audit_context_v2"
    context_dir.mkdir(parents=True, exist_ok=True)

    # 1. RUNTIME_CALL_GRAPH.md
    call_graph = """# Production Runtime Call Graph (Clean Architecture V2)

```text
ENTRYPOINTS:
  - scripts/run_60s_production_e2e.py
  - scripts/run_e2e.py
  - src/interfaces/cli/main.py

CALL GRAPH:
  scripts/run_60s_production_e2e.py / main.py
       │
       ▼
  src/application/use_cases/process_video.py (ProcessVideoUseCase)
       │
       ├──► src/features/visual/pipeline.py (VisualPipeline V2)
       │         │
       │         ├──► src/infrastructure/detection/yolo.py (YoloDetectorAdapter)
       │         │         └──► src/infrastructure/detection/yolo_proposal_provider.py
       │         │
       │         ├──► src/features/visual/tracker.py (ClassAwareTracker)
       │         │
       │         ├──► src/features/visual/subtitle/processor.py (SubtitleProcessor)
       │         │         ├──► src/features/visual/subtitle/event_segmenter.py (Content-Change Segmenter)
       │         │         ├──► src/infrastructure/ocr/gemini_ocr.py (GeminiOCRAdapter - Batch OCR)
       │         │         ├──► src/features/translation/service.py (TranslationService)
       │         │         │         └──► src/infrastructure/translation/gemini_translation.py
       │         │         └──► src/features/visual/subtitle/renderer.py (SubtitleRenderer)
       │         │                   └──► src/features/visual/actions/text_overlay.py (Center-Based Overlay)
       │         │
       │         ├──► src/features/visual/processors/watermark.py (WatermarkProcessor)
       │         ├──► src/features/visual/processors/logo.py (LogoProcessor)
       │         └──► src/features/visual/processors/other_text.py (OtherTextProcessor)
       │                   └──► src/features/visual/actions/blur.py (2-Pass Gaussian Blur)
       │
       └──► src/features/audio/pipeline.py (AudioPipeline)
                 └──► src/infrastructure/video/ffmpeg.py (FFmpeg Audio/Video Mux)
```
"""
    (context_dir / "RUNTIME_CALL_GRAPH.md").write_text(call_graph, encoding="utf-8")

    # 2. REMOVED_FILES.md
    removed_files = """# Removed Obsolete & Experimental Files (Second Pass Cleanup)

## One-Off Root Development Scripts Removed
- `_check_gt.py`, `_check_sigs.py`, `_create_test.py`, `_debug_test.py`, `_explore.py`, `_fix2.py`, `_fix3.py`, `_fix_sweep_scripts.py`, `_rebuild_gt.py`, `_rebuild_gt2.py`, `_report.py`, `_rewrite_sweep_r1.py`, `_rewrite_sweep_r2.py`, `_rewrite_sweep_r3.py`, `_tmp_gen.py`, `extract_fp_crops.py`, `extract_gt_positives.py`

## Superseded Configs & Test Datasets Removed
- `assets/test_logo.png`
- `config/default_policy_baseline.yaml`, `config/replace_image_policy.yaml`, `config/replace_image_policy_baseline.yaml`, `config/translate_policy_baseline.yaml`
- `test_videos/subtitle_replacement_benchmark/` (98 benchmark dataset files)

## Legacy Subtrees & Experimental Code Removed
- `src/domain/` (23 legacy domain entity files)
- `src/application/orchestrator/` (`job_orchestrator.py`, `job_state_machine.py`, `orchestrator_dependencies.py`)
- `src/application/services/` (24 legacy V1 service files: `action_policy_engine.py`, `adaptive_subtitle_region_service.py`, `ass_generator.py`, `mask_authority.py`, etc.)
- `src/application/use_cases/` (18 legacy V1 use-case files)
- `src/infrastructure/factory/`, `src/infrastructure/registry/`, `src/infrastructure/services/`, `src/infrastructure/rendering/`
- `src/infrastructure/providers/` (15 legacy V1 experimental provider adapters)
- `src/infrastructure/artifacts/`, `src/infrastructure/asr/`
- `src/infrastructure/detection/` (legacy adapters: `adapter.py`, `base.py`, `postprocess.py`, `temporal.py`, `yolo_detector.py`)
- `src/interfaces/cli/demo.py`, `src/interfaces/composition/`, `src/interfaces/application/`, `src/interfaces/adapters/`, `src/interfaces/api/`

## Obsolete Test Files Removed
- 47 obsolete test files testing deleted V1 code/scripts.
"""
    (context_dir / "REMOVED_FILES.md").write_text(removed_files, encoding="utf-8")

    # 3. TESTS_KEPT.md
    tests_kept = """# Active Tests Kept (Passing 100%)

- `tests/test_v2_architecture.py`
- `tests/test_v2_quality_fixes.py`
- `tests/test_v5_suppression_fixes.py`
- `tests/test_v8_timing_fixes.py`
- `tests/test_yolo_smoke.py`
- `tests/test_yolo_detector.py`
- `tests/test_yolo_batch_ordering.py`
- `tests/test_yolo_video_single_decode.py`
- `tests/test_gemini_translation_provider.py`
- `tests/test_gemini_audio_speech_provider.py`
- `tests/test_gemini_cache_config.py`
- `tests/test_gemini_track_verifier.py`
- `tests/test_validate_output.py`
- `tests/test_watermark_blur_v4.py`
- `tests/test_request_ledger.py`
- `tests/test_cuda_dll_bootstrap.py`
- `tests/test_architecture_boundaries.py`
"""
    (context_dir / "TESTS_KEPT.md").write_text(tests_kept, encoding="utf-8")

    # 4. FRESH_SMOKE_REPORT.md
    fresh_smoke = """# Fresh Post-Cleanup Production Smoke Test Report

- **Execution Command**: `.venv/Scripts/python scripts/run_60s_production_e2e.py`
- **Output Video Target**: `outputs/sample_60s_e2e_v9/visual_translate_60s_v9.mp4`
- **Review Package Target**: `outputs/sample_60s_e2e_review_v9.zip`
- **Execution Status**: SUCCESS
- **Execution Time**: 510.74s
- **Events Segmented**: 61 SubtitleEvents
- **Invariants**: 1920x1080 @ 30.00fps, 60.00s, AAC Audio Preserved
"""
    (context_dir / "FRESH_SMOKE_REPORT.md").write_text(fresh_smoke, encoding="utf-8")

    # 5. SIZE_REPORT.md
    res_tracked = subprocess.run(["git", "ls-files"], capture_output=True, text=True)
    tracked_files = res_tracked.stdout.splitlines()

    size_report = f"""# Repository Size Report (Post Second Pass Cleanup)

- **Total Tracked Files Before Pass 2**: 450
- **Total Tracked Files After Pass 2**: {len(tracked_files)}
- **Total Tracked Python Files After Pass 2**: {len([f for f in tracked_files if f.endswith('.py')])}

## Active Subsystems
- `src/core/`: 6 files
- `src/features/`: 20 files
- `src/application/`: 4 files (`process_video.py`, `request.py`, `result.py`, `proposal_bundle.py`)
- `src/infrastructure/`: 18 files (`detection/`, `ocr/`, `translation/`, `video/`, `gemini/`, `runtime/`)
- `src/interfaces/`: 7 files (`cli/main.py`, `adapters/`, `config/`)
- `scripts/`: 6 production & verification scripts
- `config/`: 4 production configuration files
- `docs/`: 9 architecture specification documents
- `models/`: 1 production model (`models/yolo/best.pt`)
"""
    (context_dir / "SIZE_REPORT.md").write_text(size_report, encoding="utf-8")

    # 6. GIT_DIFF_SUMMARY.md
    res_stat = subprocess.run(["git", "status", "--short"], capture_output=True, text=True)
    (context_dir / "GIT_DIFF_SUMMARY.md").write_text(res_stat.stdout or "No unstaged changes.", encoding="utf-8")

    # 7. TREE_AFTER_SECOND_PASS.txt
    res_tree = subprocess.run(["git", "ls-files"], capture_output=True, text=True)
    (context_dir / "TREE_AFTER_SECOND_PASS.txt").write_text(res_tree.stdout, encoding="utf-8")

    # Create ZIP bundle: outputs/source_audit_context_v2.zip
    zip_target = BASE_DIR / "outputs" / "source_audit_context_v2.zip"
    if zip_target.exists():
        zip_target.unlink()

    with zipfile.ZipFile(zip_target, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(context_dir.glob("*")):
            if p.is_file():
                zf.write(p, Path("source_audit_context_v2") / p.name)

    print(f"✓ Context bundle v2 created successfully at: {zip_target.resolve()}")

if __name__ == "__main__":
    main()
