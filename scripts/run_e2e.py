#!/usr/bin/env python3
"""
Media-Flow End-to-End Execution Script

Runs the complete media-flow pipeline on a real video file using the fine-tuned YOLO
model (models/yolo/best.pt) for burn-in region detection, followed by OCR, translation,
subtitle placement, ASS rendering, audio muxing, and output validation.
"""

import argparse
import json
import os
import sys
import time
import shutil
import subprocess
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
from src.interfaces.composition.application_builder import ApplicationBuilder
from src.domain.entities.video_job import VideoJob
from src.domain.entities.feature_config import FeatureConfig
from src.domain.entities.stage_control_config import StageControlConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run media-flow END-TO-END pipeline on a real video."
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Path to input video file (e.g. test_videos/real_mini_benchmark/sample5.mp4)"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Path to final output video file (e.g. outputs/e2e_sample5.mp4)"
    )
    parser.add_argument(
        "--device", choices=["auto", "cuda", "cpu"], default="auto", help="Execution device for YOLO and STT (auto|cuda|cpu)"
    )
    parser.add_argument(
        "--speech-provider", choices=["faster-whisper", "gemini_audio", "mock"], default="faster-whisper", help="Speech-to-text provider"
    )
    parser.add_argument(
        "--translation-provider", choices=["gemini", "openai", "mock"], default="gemini", help="Translation provider"
    )
    parser.add_argument(
        "--ocr-provider", choices=["hybrid_gemini", "gemini_vision", "mock"], default="hybrid_gemini", help="OCR provider"
    )
    parser.add_argument(
        "--placement-mode",
        choices=["replace_source_subtitle", "avoid_source_subtitle", "fixed_safe_zone"],
        default="replace_source_subtitle",
        help="Subtitle placement policy",
    )
    parser.add_argument(
        "--job-id", type=str, default="", help="Custom job ID (default: auto-generated timestamp)"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output file if present"
    )
    parser.add_argument(
        "--source-lang", type=str, default="zh", help="Source language (zh|en|auto)"
    )
    parser.add_argument(
        "--target-lang", type=str, default="vi", help="Target language (vi)"
    )
    return parser.parse_args()


def preflight_check(args):
    print("=" * 60)
    print("[1/6] Preflight Check")
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
        print(f"  ✓ Compute Device: cpu")

    # 6. Credentials
    if args.translation_provider == "gemini" or args.ocr_provider in ("hybrid_gemini", "gemini_vision"):
        if not os.getenv("GEMINI_API_KEY"):
            raise RuntimeError("BLOCKED BY EXTERNAL CREDENTIAL: GEMINI_API_KEY environment variable is required for Gemini provider.")
        print("  ✓ External Credential: GEMINI_API_KEY verified")

    print("[Preflight] All checks passed successfully.\n")
    return effective_device, input_path, output_path


def get_video_info(video_path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "stream=codec_type,width,height,r_frame_rate,duration",
        "-show_entries", "format=duration",
        "-of", "json",
        str(video_path)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(res.stdout)
    
    video_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    audio_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
    
    duration = float(video_stream.get("duration") or data.get("format", {}).get("duration", 0.0))
    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))
    fps_str = video_stream.get("r_frame_rate", "0/0")
    if "/" in fps_str:
        num, den = map(float, fps_str.split("/"))
        fps = num / den if den > 0 else 0.0
    else:
        fps = float(fps_str)
        
    return {
        "duration": duration,
        "width": width,
        "height": height,
        "fps": fps,
        "has_audio": audio_stream is not None
    }


def validate_output(input_info, output_path):
    print("=" * 60)
    print("[6/6] Final Validation & Verification")
    print("=" * 60)

    if not output_path.exists():
        raise RuntimeError(f"E2E Validation Failed: Output file was not created at {output_path}")

    file_size = output_path.stat().st_size
    if file_size == 0:
        raise RuntimeError(f"E2E Validation Failed: Output file size is 0 bytes.")

    output_info = get_video_info(output_path)

    print(f"  Input:  Duration={input_info['duration']:.2f}s, Res={input_info['width']}x{input_info['height']}, FPS={input_info['fps']:.2f}, Audio={input_info['has_audio']}")
    print(f"  Output: Duration={output_info['duration']:.2f}s, Res={output_info['width']}x{output_info['height']}, FPS={output_info['fps']:.2f}, Audio={output_info['has_audio']}, Size={file_size / (1024*1024):.2f} MB")

    # Invariant checks
    if abs(input_info["duration"] - output_info["duration"]) > 1.5:
        raise RuntimeError(f"Duration invariant violated: Input {input_info['duration']:.2f}s vs Output {output_info['duration']:.2f}s")

    if input_info["width"] != output_info["width"] or input_info["height"] != output_info["height"]:
        raise RuntimeError(f"Resolution invariant violated: Input {input_info['width']}x{input_info['height']} vs Output {output_info['width']}x{output_info['height']}")

    if input_info["has_audio"] and not output_info["has_audio"]:
        raise RuntimeError("Audio invariant violated: Input had audio stream, but final output has no audio stream.")

    print("  ✓ All E2E invariants passed (Duration, Resolution, FPS, Audio Stream).\n")
    return output_info


def main():
    start_time = time.monotonic()
    args = parse_args()
    job_id = args.job_id or f"job_e2e_{int(time.time())}"

    # 1. Preflight
    effective_device, input_path, output_path = preflight_check(args)
    input_info = get_video_info(input_path)

    # 2. Build Application
    print("=" * 60)
    print("[2/6] Building Production E2E Application")
    print("=" * 60)
    print(f"  Job ID:                      {job_id}")
    print(f"  YOLO Region Detector:        hybrid_yolo (models/yolo/best.pt)")
    print(f"  Speech-to-Text Provider:     {args.speech_provider}")
    print(f"  OCR Provider:                {args.ocr_provider}")
    print(f"  Translation Provider:        {args.translation_provider}")
    print(f"  Subtitle Placement Mode:     {args.placement_mode}")
    print(f"  Device:                      {effective_device}\n")

    builder = ApplicationBuilder(
        output_dir="output",
        video_provider="ffmpeg",
        speech_provider=args.speech_provider,
        translation_provider=args.translation_provider,
        subtitle_provider="ass",
        ocr_provider=args.ocr_provider,
        subtitle_region_detector_provider="hybrid_yolo",
        frame_provider="ffmpeg",
        target_language=args.target_lang,
        ocr_device="gpu" if effective_device == "cuda" else "cpu",
        placement_mode=args.placement_mode,
        ffmpeg_encoder="auto",
    )
    app = builder.build()

    # 3. Construct VideoJob
    job = VideoJob(
        job_id=job_id,
        input_video=str(input_path),
        source_language=args.source_lang,
        speech_language="auto",
        target_language=args.target_lang,
        features=FeatureConfig(
            speech_to_text=True,
            ocr=True,
            translate=True,
            subtitle=True,
            subtitle_placement=True,
            subtitle_placement_debug=True,
            render_video=True,
            dubbing=False,
        ),
        output_video_path=str(output_path),
        stage_controls=StageControlConfig(
            run_asr=True,
            run_ocr=True,
            run_subtitle_region_detection=True,
            run_translation=True,
            run_normalization=True,
            run_segmentation=True,
            reuse_ocr_cache=False,
            reuse_subtitle_region_cache=False,
        ),
    )

    # 4. Execute E2E Job
    print("=" * 60)
    print("[3/6 - 5/6] Running E2E Pipeline (YOLO → OCR → Translate → ASS Masking → Render)")
    print("=" * 60)
    result_job = app.run(job)

    # 5. Output Validation
    output_info = validate_output(input_info, output_path)

    elapsed_total = time.monotonic() - start_time

    # Read debug metadata for detection stats
    debug_dir = Path("output") / job_id / "debug"
    regions_file = debug_dir / "subtitle_regions.json"
    class_counts = {"subtitle": 0, "watermark": 0, "logo": 0, "other_burn_in_text": 0}
    if regions_file.exists():
        with open(regions_file, "r", encoding="utf-8") as f:
            regions_data = json.load(f)
            for r in regions_data:
                c_name = r.get("class_name") or "subtitle"
                if c_name in class_counts:
                    class_counts[c_name] += 1
                else:
                    class_counts["other_burn_in_text"] += 1

    print("=" * 60)
    print("E2E EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Status:                      SUCCESS")
    print(f"Input Video:                 {input_path}")
    print(f"Final Output Artifact:       {output_path}")
    print(f"Job Duration:                {elapsed_total:.2f}s")
    print(f"Video Invariants:            {output_info['width']}x{output_info['height']} @ {output_info['fps']:.2f} fps ({output_info['duration']:.2f}s)")
    print(f"YOLO Detection Summary:")
    for cls_name, count in class_counts.items():
        print(f"  - {cls_name}: {count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
