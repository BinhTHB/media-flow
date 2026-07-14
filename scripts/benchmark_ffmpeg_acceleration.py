import argparse
import json
import subprocess
import time
from pathlib import Path

from src.domain.entities.resource_policy import FFmpegConfig
from src.infrastructure.runtime.ffmpeg_capability_detector import (
    detect_ffmpeg_capabilities,
)
from src.infrastructure.runtime.ffmpeg_encode import build_ffmpeg_encode_args


def _duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        capture_output=True, check=True, text=True,
    )
    return float(result.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("output/ffmpeg_benchmark"))
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    capabilities = detect_ffmpeg_capabilities()
    source_duration = _duration(args.video)
    variants = [
        ("cpu", FFmpegConfig(encoder="cpu")),
        ("nvenc", FFmpegConfig(encoder="nvenc")),
    ]
    if capabilities.cuda_hwaccel_available:
        variants.append(
            ("cuda_decode_nvenc", FFmpegConfig(
                encoder="nvenc", decoder="cuda", use_gpu_decode=True
            ))
        )
    results = []
    for name, config in variants:
        if name != "cpu" and not capabilities.nvenc_available:
            continue
        output = args.output_root / f"{name}.mp4"
        plan = build_ffmpeg_encode_args(
            str(args.video), str(output), config, False, capabilities
        )
        started = time.monotonic()
        subprocess.run(plan.args, check=True, capture_output=True, text=True)
        elapsed = time.monotonic() - started
        results.append({
            "mode": name,
            "encoder": plan.encoder_used,
            "gpu_decode_used": plan.gpu_decode_used,
            "elapsed_time_seconds": round(elapsed, 3),
            "output_size_bytes": output.stat().st_size,
            "duration_seconds": round(_duration(output), 3),
            "duration_preserved": abs(_duration(output) - source_duration) < 0.1,
        })
    recommendation = min(results, key=lambda item: item["elapsed_time_seconds"])["mode"]
    report = {
        "capabilities": capabilities.to_dict(),
        "source_duration_seconds": source_duration,
        "results": results,
        "selected_recommendation": recommendation,
    }
    report_path = args.output_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
