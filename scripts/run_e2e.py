#!/usr/bin/env python3
"""
Media-Flow End-to-End Runner (Policy-Driven Architecture)

Executes the unified production pipeline:
YOLO 4-class Detection -> Class-Aware Tracker -> Action Policy Engine -> RenderingPlan -> Lazy Enrichment -> Video Renderer -> Output Validation.
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
from src.application.dto.processing_request import ProcessingRequest
from src.infrastructure.services.media_processing_service import MediaProcessingService
from src.domain.entities.processing_policy import ProcessingPolicyConfig, get_preset_policy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run policy-driven media-flow END-TO-END pipeline on a video."
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Path to input video file (e.g. test_videos/real_mini_benchmark/sample5.mp4)"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Path to final output video file (e.g. outputs/e2e_sample5.mp4)"
    )
    parser.add_argument(
        "--preset",
        choices=["default", "blur-all", "translate-subtitle", "preserve-all"],
        default="default",
        help="Pre-built policy preset (default: default)",
    )
    parser.add_argument(
        "--policy", type=str, default="", help="Path to custom policy YAML configuration file"
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

    # 6. Credentials check only if policy requires replace_text overlay
    policy = get_preset_policy(args.preset) if not args.policy else ProcessingPolicyConfig.from_yaml_file(args.policy)
    needs_gemini = False
    for class_cfg in (policy.subtitle, policy.watermark, policy.logo, policy.other_burn_in_text):
        for o in class_cfg.overlays:
            if o.action == "replace_text":
                needs_gemini = True

    if needs_gemini:
        if not os.getenv("GEMINI_API_KEY"):
            raise RuntimeError("BLOCKED BY EXTERNAL CREDENTIAL: GEMINI_API_KEY environment variable is required for replace_text overlay action.")
        print("  ✓ External Credential: GEMINI_API_KEY verified")
    else:
        print("  ✓ Offline Execution: Policy does not require Gemini API. GEMINI_API_KEY check bypassed.")

    print("[Preflight] All checks passed successfully.\n")
    return effective_device, input_path, output_path, policy


def main():
    args = parse_args()
    effective_device, input_path, output_path, policy = preflight_check(args)
    job_id = args.job_id or f"job_policy_{int(time.time())}"

    request = ProcessingRequest(
        job_id=job_id,
        input_video=str(input_path),
        output_video=str(output_path),
        device=effective_device,
        preset=args.preset,
        policy_config=policy if not args.policy else None,
        policy_file=args.policy if args.policy else None,
        overwrite=args.overwrite or True,
    )

    print("=" * 60)
    print("[2/5 - 5/5] Running Policy-Driven Media Processing Pipeline")
    print("=" * 60)
    print(f"  Preset / Policy: {args.preset if not args.policy else args.policy}")
    print(f"  Class Policies:")
    print(f"    - Subtitle:           bg={policy.subtitle.background.action}, overlays={[o.action for o in policy.subtitle.overlays]}")
    print(f"    - Watermark:          bg={policy.watermark.background.action}, overlays={[o.action for o in policy.watermark.overlays]}")
    print(f"    - Logo:               bg={policy.logo.background.action}, overlays={[o.action for o in policy.logo.overlays]}")
    print(f"    - Other Burn-in Text: bg={policy.other_burn_in_text.background.action}, overlays={[o.action for o in policy.other_burn_in_text.overlays]}\n")

    service = MediaProcessingService()
    result = service.process_video(request)

    print("=" * 60)
    print("E2E EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Status:                      {result.status}")
    print(f"Input Video:                 {result.input_video_path}")
    print(f"Final Output Artifact:       {result.output_video_path}")
    print(f"Action Plan Artifact:        {result.action_plan_path}")
    print(f"Total Duration:              {result.timings.get('total', 0.0):.2f}s")
    print(f"Video Info:                  {result.resolution} @ {result.fps:.2f} fps ({result.duration_sec:.2f}s)")
    print(f"YOLO Detection Summary:")
    for cls_name, count in result.detections_summary.items():
        print(f"  - {cls_name}: {count}")
    print(f"Actions Executed:")
    for act_type, count in result.actions_summary.items():
        print(f"  - {act_type}: {count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
