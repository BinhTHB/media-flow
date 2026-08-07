#!/usr/bin/env python3
"""
MediaFlow End-to-End Runner (Feature-First Modular Clean Architecture V2)

Executes video processing via ProcessVideoUseCase, AudioPipeline, and VisualPipeline.
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Load .env file automatically if present
env_path = BASE_DIR / ".env"
if env_path.exists():
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k not in os.environ:
                    os.environ[k] = v

import torch
from src.application.process_video import ProcessVideoUseCase
from src.application.request import ProcessingRequest
from src.features.audio.pipeline import AudioPipeline, AudioRequest
from src.features.translation.service import TranslationService
from src.features.visual.pipeline import VisualPipeline, VisualRequest
from src.features.visual.subtitle.processor import SubtitleProcessor
from src.infrastructure.asr.whisper import WhisperASRAdapter
from src.infrastructure.detection.yolo import YoloDetectorAdapter
from src.infrastructure.translation.gemini_translation import GeminiTranslationAdapter
from src.infrastructure.video.ffmpeg import FFmpegVideoAdapter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run feature-first modular MediaFlow END-TO-END pipeline on a video."
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Path to input video file (e.g. test_videos/real_mini_benchmark/sample5.mp4)"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Path to final output video file (e.g. outputs/e2e_sample5.mp4)"
    )
    parser.add_argument(
        "--preset",
        choices=["blur-all", "translate-subtitle", "preserve-all"],
        default="blur-all",
        help="Policy preset (default: blur-all)",
    )
    parser.add_argument(
        "--policy", type=str, default="", help="Path to policy YAML file"
    )
    parser.add_argument(
        "--device", choices=["auto", "cuda", "cpu"], default="auto", help="Execution device (auto|cuda|cpu)"
    )
    parser.add_argument(
        "--job-id", type=str, default="", help="Custom job ID"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output file if present"
    )
    return parser.parse_args()


def preflight_check(args):
    print("=" * 60)
    print("[1/5] Preflight Check")
    print("=" * 60)

    # 1. Input video
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input video does not exist: {input_path}")
    print(f"  ✓ Input video: {input_path}")

    # 2. YOLO weights
    weights_path = BASE_DIR / "models" / "yolo" / "best.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO weights not found at expected location: {weights_path}")
    print(f"  ✓ YOLO weights: {weights_path}")

    # 3. FFmpeg and FFprobe
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg executable not found in PATH.")
    if not shutil.which("ffprobe"):
        raise RuntimeError("FFprobe executable not found in PATH.")
    print("  ✓ FFmpeg / FFprobe: Available")

    # 4. Output directory
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output path already exists (use --overwrite to replace): {output_path}")
    print(f"  ✓ Output directory: {output_path.parent} (writable)")

    # 5. Device check
    if args.device in ("cuda", "auto"):
        cuda_ok = torch.cuda.is_available()
        effective_device = "cuda" if cuda_ok else "cpu"
        if args.device == "cuda" and not cuda_ok:
            raise RuntimeError("CUDA requested via --device cuda but CUDA is not available.")
        print(f"  ✓ Compute Device: {effective_device} (Requested: {args.device}, CUDA available: {cuda_ok})")
    else:
        effective_device = "cpu"
        print("  ✓ Compute Device: cpu")

    # 6. Credentials check only if preset/policy requires translate-subtitle
    needs_gemini = args.preset == "translate-subtitle" or (args.policy and "translate" in args.policy)
    if needs_gemini:
        if not os.getenv("GEMINI_API_KEY"):
            raise RuntimeError("BLOCKED BY EXTERNAL CREDENTIAL: GEMINI_API_KEY environment variable is required for subtitle translation.")
        print("  ✓ External Credential: GEMINI_API_KEY verified")
    else:
        print("  ✓ Offline Execution: Policy does not require Gemini API. GEMINI_API_KEY check bypassed.")

    print("[Preflight] All checks passed successfully.\n")
    return effective_device, input_path, output_path


def main():
    args = parse_args()
    effective_device, input_path, output_path = preflight_check(args)
    job_id = args.job_id or f"job_v2_{int(time.time())}"

    # Build requests based on preset
    audio_req = AudioRequest(enabled=False)
    if args.preset == "translate-subtitle":
        visual_req = VisualRequest(
            enabled=True,
            subtitle_enabled=True,
            subtitle_mode="translate_replace",
            logo_enabled=True,
            logo_action="preserve",
            watermark_enabled=True,
            watermark_action="preserve",
            other_text_enabled=True,
            other_text_action="preserve",
        )
    elif args.preset == "preserve-all":
        visual_req = VisualRequest(
            enabled=True,
            subtitle_enabled=True,
            subtitle_mode="preserve",
            logo_enabled=True,
            logo_action="preserve",
            watermark_enabled=True,
            watermark_action="preserve",
            other_text_enabled=True,
            other_text_action="preserve",
        )
    else:  # blur-all / default
        visual_req = VisualRequest(
            enabled=True,
            subtitle_enabled=True,
            subtitle_mode="blur",
            logo_enabled=True,
            logo_action="blur",
            watermark_enabled=True,
            watermark_action="blur",
            other_text_enabled=True,
            other_text_action="blur",
        )

    request = ProcessingRequest(
        job_id=job_id,
        input_video=str(input_path),
        output_video=str(output_path),
        audio=audio_req,
        visual=visual_req,
        device=effective_device,
        overwrite=args.overwrite or True,
    )

    print("=" * 60)
    print("[2/5 - 5/5] Running Modular Clean Architecture V2 Pipeline")
    print("=" * 60)
    print(f"  Preset: {args.preset}")
    print(f"  Visual Subtitle Mode: {visual_req.subtitle_mode}\n")

    # Wire Providers and Feature Pipelines
    video_util = FFmpegVideoAdapter()
    detector = YoloDetectorAdapter(device=effective_device)
    
    ocr_provider = None
    translation_service = None
    if args.preset == "translate-subtitle":
        from src.infrastructure.ocr.gemini_ocr import GeminiOCRAdapter
        ocr_provider = GeminiOCRAdapter()
        translation_adapter = GeminiTranslationAdapter()
        translation_service = TranslationService(translation_adapter)

    subtitle_processor = SubtitleProcessor(ocr_provider=ocr_provider, translation_service=translation_service)
    visual_pipeline = VisualPipeline(detector=detector, subtitle_processor=subtitle_processor)

    use_case = ProcessVideoUseCase(visual_pipeline=visual_pipeline, video_util=video_util)
    result = use_case.execute(request)

    print("=" * 60)
    print("E2E EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Status:                      {result.status}")
    print(f"Input Video:                 {result.input_video_path}")
    print(f"Final Output Artifact:       {result.output_video_path}")
    print(f"Workspace Directory:         {result.workspace_dir}")
    print(f"Total Duration:              {result.timings.get('total', 0.0):.2f}s")
    print(f"Video Info:                  {result.resolution} @ {result.fps:.2f} fps ({result.duration_sec:.2f}s)")
    print(f"YOLO Detection Summary:")
    for cls_name, count in result.detections_summary.items():
        print(f"  - {cls_name}: {count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
