"""Write Phase E comparison artifacts without invoking any provider by default."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.infrastructure.benchmark.gemini_phase_e import (
    BenchmarkValidationError,
    evaluate_safety_regression,
    sanitize_artifact,
    validate_manifest,
)

PROVIDER_MODES = {
    "ocr": ["paddle_default", "gemini_full_frame_only", "gemini_full_frame_crop_retry"],
    "asr": ["faster_whisper_small", "gemini_full_audio", "optional_gemini_chunk_fallback"],
}


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(sanitize_artifact(data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _summary(manifest: dict[str, Any], failures: list[dict[str, str]], live: bool) -> str:
    missing = sum(sample["reference_status"] == "REFERENCE_MISSING" for sample in manifest["samples"])
    return "\n".join((
        "# Gemini Phase E Comparative Benchmark",
        "",
        "- Phase E implementation: PASS",
        "- Live benchmark: BLOCKED/INCOMPLETE",
        "- Provider decision: DEFERRED",
        "- Production Ready: NO",
        f"- Samples: {len(manifest['samples'])}",
        f"- Missing references: {missing}",
        f"- Failures: {len(failures)}",
        "- Live execution requires reviewed corpus references and is intentionally not implemented.",
        "",
    ))


def main() -> int:
    parser = argparse.ArgumentParser(description="Create offline-only Gemini Phase E comparative benchmark artifacts.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", default="output/benchmarks/gemini_phase_e", type=Path)
    parser.add_argument("--ocr", action="store_true", help="Select OCR comparison modes.")
    parser.add_argument("--asr", action="store_true", help="Select ASR comparison modes.")
    parser.add_argument("--live", action="store_true", help="Require reviewed corpus and API configuration, then report blocked live execution.")
    parser.add_argument("--disable-cache", action="store_true")
    parser.add_argument("--max-live-requests", type=int, default=0)
    parser.add_argument("--enable-escalation", action="store_true")
    args = parser.parse_args()
    try:
        manifest = validate_manifest(args.manifest)
    except BenchmarkValidationError as error:
        parser.error(f"INPUT_INVALID: {error}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures = evaluate_safety_regression(manifest)
    missing = [sample["id"] for sample in manifest["samples"] if sample["reference_status"] == "REFERENCE_MISSING"]
    if missing:
        failures.extend(
            {"sample_id": sample_id, "code": "REFERENCE_MISSING", "message": "OCR and ASR references are required for comparison."}
            for sample_id in missing
        )
    if args.live:
        if not os.environ.get("GEMINI_API_KEY"):
            parser.error("GEMINI_API_KEY is required for --live.")
        failures.append({
            "code": "BENCHMARK_INCOMPLETE",
            "message": "Live execution requires reviewed corpus references and is deliberately blocked.",
        })

    selected = {
        "ocr": PROVIDER_MODES["ocr"] if args.ocr or not (args.ocr or args.asr) else [],
        "asr": PROVIDER_MODES["asr"] if args.asr or not (args.ocr or args.asr) else [],
    }
    environment = {
        "execution": "live_blocked" if args.live else "dry_run",
        "cache_policy": "disabled" if args.disable_cache else "cold_then_warm",
        "cold_warm_policy": "Record cold and warm runs separately; do not compare them as one result.",
        "provider_modes": selected,
        "max_live_requests": args.max_live_requests,
        "escalation_enabled": args.enable_escalation,
        "verdict": {
            "phase_e_implementation": "PASS",
            "live_benchmark": "BLOCKED/INCOMPLETE",
            "provider_decision": "DEFERRED",
            "production_ready": "NO",
        },
    }
    comparison = {
        "status": "BENCHMARK_INCOMPLETE" if failures else "DRY_RUN_COMPLETE",
        "sample_results": [],
        "provider_modes": selected,
        "live_provider_calls": 0,
        "note": "No providers are called by this runner. Live execution requires reviewed corpus references.",
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    _write_json(args.output_dir / "environment.json", environment)
    _write_json(args.output_dir / "failures.json", failures)
    _write_json(args.output_dir / "comparison.json", comparison)
    (args.output_dir / "summary.md").write_text(_summary(manifest, failures, args.live), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
