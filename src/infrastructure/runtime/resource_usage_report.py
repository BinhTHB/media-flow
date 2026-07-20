from __future__ import annotations

import json
from pathlib import Path

from src.domain.entities.resource_policy import ResourcePolicy
from src.infrastructure.runtime.ffmpeg_capability_detector import (
    FFmpegCapabilityReport,
)
from src.infrastructure.runtime.ffmpeg_encode import FFmpegEncodePlan


def write_resource_usage_report(
    output_root: Path,
    policy: ResourcePolicy,
    capabilities: FFmpegCapabilityReport,
    encode_plan: FFmpegEncodePlan,
    *,
    ocr_device: str = "not_used",
    asr_device: str = "not_used",
    expensive_steps_enabled: list[str] | None = None,
    expensive_steps_skipped: list[str] | None = None,
) -> Path:
    skipped = expensive_steps_skipped or []
    report = {
        "resource_profile": policy.profile,
        "ffmpeg_encoder_used": encode_plan.encoder_used,
        "gpu_decode_used": encode_plan.gpu_decode_used,
        "nvenc_available": capabilities.nvenc_available,
        "ocr_device": ocr_device,
        "asr_device": asr_device,
        "expensive_steps_enabled": expensive_steps_enabled or [],
        "expensive_steps_skipped": skipped,
        "reason_for_skipping_heavy_algorithms": (
            "Disproportionate compute or unsafe automatic replacement."
            if skipped else None
        ),
    }
    path = output_root / "debug" / "resource_usage_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
