import argparse
from collections import Counter
import json
import shutil
import subprocess
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw

from scripts.render_adaptive_subtitles import _target_text
from src.application.services.ass_generator import ASSGenerator
from src.application.services.subtitle_render_policy import SubtitleRenderPolicy
from src.domain.entities.resource_policy import FFmpegConfig, ResourcePolicy
from src.domain.entities.segment import Segment
from src.domain.entities.subtitle_placement import SubtitlePlacement
from src.domain.entities.subtitle_style import SubtitleStyle
from src.infrastructure.providers.subtitle.ass_subtitle_provider import ASSSubtitleProvider
from src.infrastructure.providers.video.ffmpeg_video_provider import FFmpegVideoProvider
from src.infrastructure.runtime.ffmpeg_capability_detector import detect_ffmpeg_capabilities
from src.infrastructure.runtime.ffmpeg_encode import build_ffmpeg_encode_args
from src.infrastructure.runtime.resource_usage_report import write_resource_usage_report


def _segment(item: dict) -> Segment:
    return Segment(
        item["id"], float(item["start"]), float(item["end"]), item["text"],
        item.get("translated_text", ""), item["source"], float(item["confidence"]),
    )


def _interval_for(segment: Segment, intervals: list[dict]) -> dict | None:
    return next(
        (
            item for item in intervals
            if item["timestamp_start"] <= segment.start < item["timestamp_end"]
        ),
        None,
    )


def _wrap(text: str) -> str:
    return "\n".join(textwrap.wrap(text, width=42)[:2])


def _contact_sheet(video: Path, output: Path, title: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for timestamp in (5, 10, 15):
        path = output.parent / f"{output.stem}_{timestamp}s.jpg"
        subprocess.run(
            [
                "ffmpeg", "-ss", str(timestamp), "-i", str(video),
                "-frames:v", "1", "-q:v", "2", "-y", str(path),
            ],
            check=True, capture_output=True, text=True,
        )
        frames.append(Image.open(path).convert("RGB"))
    width = frames[0].width
    header = 48
    sheet = Image.new("RGB", (width * len(frames), frames[0].height + header), "white")
    ImageDraw.Draw(sheet).text((12, 15), title, fill="black")
    for index, frame in enumerate(frames):
        sheet.paste(frame, (index * width, header))
    sheet.save(output)

def _video_duration(video: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video),
        ],
        capture_output=True, text=True,
    )
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        return 0.0
    return duration if result.returncode == 0 and duration > 0 else 0.0

def _timing_is_valid(segments: list[Segment], duration: float) -> bool:
    return bool(segments) and duration > 0 and all(
        0 <= segment.start < segment.end <= duration + 0.1 for segment in segments
    )

def _static_style(name: str, position: str) -> tuple[SubtitleStyle, SubtitlePlacement]:
    alignment, margin_v = {
        "bottom": (2, 70),
        "upper_bottom": (2, 150),
        "top_safe": (8, 70),
    }[position]
    background, outline_width = {
        "box": ("box", 2.0),
        "outline": ("outline", 3.0),
        "minimal": ("none", 0.0),
    }[name]
    style = SubtitleStyle(
        font_size=36, outline_width=outline_width, alignment=alignment,
        margin_v=margin_v, background=background,
    )
    return style, SubtitlePlacement(
        alignment=alignment, margin_v=margin_v, is_fallback=True
    )

def _write_sidecars(
    root: Path, segments: list[Segment], style: SubtitleStyle,
    placements: list[SubtitlePlacement],
) -> dict[str, str]:
    sidecar_dir = root / "subtitles"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    provider = ASSSubtitleProvider()
    srt_path = sidecar_dir / "output.srt"
    ass_path = sidecar_dir / "output.ass"
    srt_path.write_text(provider.generate_srt(segments), encoding="utf-8")
    ass_path.write_text(
        provider.generate_ass(segments, style, placements), encoding="utf-8"
    )
    return {"srt": str(srt_path), "ass": str(ass_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--region-report", type=Path, required=True)
    parser.add_argument("--interval-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("output/subtitle_policy_v6"))
    parser.add_argument("--legacy-adaptive-report", type=Path)
    parser.add_argument("--sample", required=True)
    parser.add_argument(
        "--subtitle-output-mode",
        choices=(
            "source_replace", "static_subtitle_burn", "dual_output",
            "adaptive_source_replace", "full_review_pack",
        ),
        default="dual_output",
    )
    parser.add_argument("--force-static-subtitle-burn", action="store_true")
    parser.add_argument(
        "--static-subtitle-position",
        choices=("bottom", "upper_bottom", "top_safe"),
        default="bottom",
    )
    parser.add_argument(
        "--static-subtitle-style",
        choices=("box", "outline", "minimal"),
        default="outline",
    )
    parser.add_argument(
        "--resource-profile", choices=("low", "balanced", "quality"),
        default="balanced",
    )
    parser.add_argument("--ffmpeg-encoder", choices=("auto", "cpu", "nvenc"), default="auto")
    args = parser.parse_args()

    policy = ResourcePolicy(args.resource_profile)
    ffmpeg_config = FFmpegConfig(encoder=args.ffmpeg_encoder)
    interval_report = json.loads(args.interval_report.read_text(encoding="utf-8"))
    intervals = interval_report["intervals"]
    video_decision = interval_report.get("video_decision")
    source_segments = [_segment(item) for item in json.loads(args.captions.read_text(encoding="utf-8"))]
    decisions = []
    adaptive_decisions = []
    fallback_segments = []
    text_sources = Counter()
    render_policy = SubtitleRenderPolicy()
    for segment in source_segments:
        interval = _interval_for(segment, intervals)
        text, source, _, missing = _target_text(segment, interval, "normalized")
        decision = render_policy.decide(
            interval, has_target_text=not missing,
            resource_profile=policy.profile,
        )
        if missing and interval:
            candidate = interval.get("selected_candidate") or {}
            text = candidate.get("raw_ocr_text", "")
            source = "ocr" if text else None
            decision = render_policy.decide(
                interval, has_target_text=bool(text),
                resource_profile=policy.profile,
            )
        decisions.append({
            "segment_id": segment.id,
            "decision": decision.decision,
            "reason": decision.reason,
            "target_text_source": source,
        })
        adaptive = render_policy.adaptive_decide(
            interval, has_target_text=bool(text), video_decision=video_decision
        )
        override = render_policy.adaptive_override_evidence(
            interval, has_target_text=bool(text), video_decision=video_decision
        )
        adaptive_decisions.append({
            "segment_id": segment.id,
            "interval_id": (interval or {}).get("interval_id"),
            "original_decision": (interval or {}).get("decision"),
            "original_reason": (
                (interval or {}).get("reason")
                or (interval or {}).get("review_reason")
                or (interval or {}).get("skip_reason")
            ),
            "decision": adaptive.decision,
            "reason": adaptive.reason,
            "candidate_override_eligible": override["eligible"],
            "candidate_override_applied": override["applied"],
            "candidate_override_reason": override["reason"],
            "selected_candidate_bbox": override.get("selected_candidate_bbox"),
            "local_candidate_confidence": override.get("local_candidate_confidence"),
            "target_text_present": override.get("target_text_present", bool(text)),
            "mask_area_ratio": override.get("mask_area_ratio"),
            "scene_text_risk_score": override.get("scene_text_risk_score"),
            "menu_signage_risk_score": override.get("menu_signage_risk_score"),
            "wall_sign_risk_score": override.get("wall_sign_risk_score"),
            "watermark_risk_score": override.get("watermark_risk_score"),
            "person_risk_score": override.get("person_risk_score"),
        })
        if text:
            fallback_segments.append(Segment(
                segment.id, segment.start, segment.end, segment.text,
                _wrap(text), segment.source, segment.confidence,
            ))
            text_sources[source or "unknown"] += 1

    auto_count = sum(item["decision"] == "AUTO_SOURCE_REPLACE" for item in decisions)
    adaptive_count = sum(
        item["decision"] == "AUTO_SOURCE_REPLACE" for item in adaptive_decisions
    )
    review_count = sum(item["decision"] == "REVIEW_REQUIRED" for item in decisions)
    skip_count = sum(item["decision"] == "SKIP_REPLACEMENT" for item in decisions)
    target_timeline_exists = bool(fallback_segments)
    duration = _video_duration(args.video)
    timing_valid = _timing_is_valid(fallback_segments, duration)
    includes_replacement = args.subtitle_output_mode in {
        "source_replace", "dual_output", "adaptive_source_replace", "full_review_pack",
    }
    includes_static = (
        args.subtitle_output_mode in {
            "static_subtitle_burn", "dual_output", "full_review_pack",
        }
        or (
            args.resource_profile == "balanced"
            and includes_replacement
        )
        or args.force_static_subtitle_burn
    )
    replacement_count = (
        adaptive_count
        if args.subtitle_output_mode in {"adaptive_source_replace", "full_review_pack"}
        else auto_count
    )
    replacement_gate = render_policy.replacement_gate(
        confident_interval_count=replacement_count,
        output_mode_includes_replacement=includes_replacement,
    )
    static_gate = render_policy.static_render_gate(
        target_timeline_exists=target_timeline_exists,
        timing_valid=timing_valid,
        output_mode_includes_static=includes_static,
    )
    sidecar_gate = render_policy.sidecar_generation_gate(
        target_timeline_exists=target_timeline_exists
    )
    static_allowed = static_gate.decision == "STATIC_SUBTITLE_RENDER"
    duplicate_subtitle_risk = any(
        interval.get("source_subtitle_present_score", 0.0) >= 0.70
        for interval in intervals
    )
    source_root = args.output_root / f"{args.sample}_adaptive_source_replace"
    strict_root = args.output_root / f"{args.sample}_strict_source_replace"
    static_root = args.output_root / f"{args.sample}_static_subtitle_burn"
    sidecar_root = args.output_root / f"{args.sample}_sidecar"
    capabilities = detect_ffmpeg_capabilities()

    if args.subtitle_output_mode == "full_review_pack":
        strict_video = strict_root / "videos" / "output.mp4"
        strict_video.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.video, strict_video)
        strict_debug = strict_root / "debug"
        strict_debug.mkdir(parents=True, exist_ok=True)
        (strict_debug / "strict_source_replace_report.json").write_text(
            json.dumps(
                {
                    "source_replace_count": auto_count,
                    "artifact_type": (
                        "strict_source_replace"
                        if auto_count
                        else "source_passthrough_safety_fallback"
                    ),
                    "reason": (
                        None
                        if auto_count
                        else "no_high_confidence_source_replace_intervals"
                    ),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    if includes_replacement:
        source_video = source_root / "videos" / "output.mp4"
        source_video.parent.mkdir(parents=True, exist_ok=True)
        if replacement_gate.decision == "AUTO_SOURCE_REPLACE":
            subprocess.run([
                str(Path(".venv/Scripts/python.exe")),
                "-m",
                "scripts.render_adaptive_subtitles",
                "--video", str(args.video),
                "--captions", str(args.captions),
                "--region-report", str(args.region_report),
                "--interval-report", str(args.interval_report),
                "--output-root", str(source_root),
                "--coverage-mode",
                (
                    "adaptive_source_replace"
                    if args.subtitle_output_mode in {
                        "adaptive_source_replace", "full_review_pack",
                    }
                    else "safe_replace"
                ),
                "--resource-profile", args.resource_profile,
                "--ffmpeg-encoder", args.ffmpeg_encoder,
            ], check=True)
        else:
            plan = build_ffmpeg_encode_args(
                str(args.video), str(source_video), ffmpeg_config, False, capabilities
            )
            subprocess.run(plan.args, check=True, capture_output=True, text=True)
            write_resource_usage_report(
                source_root, policy, capabilities, plan,
                expensive_steps_skipped=["dynamic_subtitle_replacement"],
            )
        _contact_sheet(
            source_video,
            source_root / "review_frames"
            / "contact_sheet_adaptive_source_replace_output.jpg",
            "ADAPTIVE SOURCE REPLACEMENT - no bottom fallback",
        )
    sidecars = {}
    if sidecar_gate.decision == "GENERATE_SIDECARS":
        style, placement = _static_style(
            args.static_subtitle_style, args.static_subtitle_position
        )
        sidecars = _write_sidecars(
            sidecar_root, fallback_segments, style, [placement] * len(fallback_segments)
        )
    static_generated = False
    static_render_error = None
    if static_allowed:
        static_video = static_root / "videos" / "output.mp4"
        static_video.parent.mkdir(parents=True, exist_ok=True)
        subtitle_path = sidecar_root / "subtitles" / "output.ass"
        style, placement = _static_style(
            args.static_subtitle_style, args.static_subtitle_position
        )
        placements = [placement for _ in fallback_segments]
        if not subtitle_path.is_file():
            subtitle_path.parent.mkdir(parents=True, exist_ok=True)
            subtitle_path.write_text(
                ASSGenerator.generate(fallback_segments, style, placements),
                encoding="utf-8",
            )
        try:
            FFmpegVideoProvider(
                ffmpeg_config=ffmpeg_config, resource_policy=policy
            ).render_video(
                str(args.video.resolve()), str(subtitle_path.resolve()),
                str(static_video.resolve()),
            )
            static_generated = True
            _contact_sheet(
                static_video,
                static_root / "review_frames"
                / "contact_sheet_static_subtitle_burn_output.jpg",
                "STATIC SUBTITLE BURN - not source replacement",
            )
        except Exception as error:
            static_render_error = str(error)

    report = {
        "layout_mode": interval_report.get("layout", {}).get("layout_mode", "unknown"),
        "output_mode": args.subtitle_output_mode,
        "resource_profile": args.resource_profile,
        "source_replace_allowed": replacement_gate.decision == "AUTO_SOURCE_REPLACE",
        "source_replace_count": auto_count,
        "adaptive_source_replace_count": adaptive_count,
        "adaptive_source_replace_artifact_type": (
            "adaptive_source_replace"
            if replacement_gate.decision == "AUTO_SOURCE_REPLACE"
            else "source_passthrough_safety_fallback"
        ),
        "static_subtitle_render_allowed": static_allowed,
        "static_subtitle_render_count": len(fallback_segments) if static_generated else 0,
        "target_timeline_exists": target_timeline_exists,
        "duplicate_subtitle_risk": duplicate_subtitle_risk,
        "static_render_blocked_reason": (
            "render_error" if static_render_error else static_gate.reason
        ),
        "replacement_blocked_reason": replacement_gate.reason,
        "output_type": (
            "full_review_pack"
            if args.subtitle_output_mode == "full_review_pack"
            else "adaptive_source_replace"
            if replacement_gate.decision == "AUTO_SOURCE_REPLACE"
            else "static_subtitle_burn_not_source_replacement"
        ),
        "review_required_count": review_count,
        "skip_count": skip_count,
        "target_text_source_distribution": dict(text_sources),
        "static_subtitle_generated": static_generated,
        "static_render_error": static_render_error,
        "replacement_gate_result": {
            "decision": replacement_gate.decision,
            "reason": replacement_gate.reason,
            "artifact_type": (
                "adaptive_source_replace"
                if replacement_gate.decision == "AUTO_SOURCE_REPLACE"
                else "source_passthrough_safety_fallback"
                if includes_replacement
                else None
            ),
        },
        "static_render_gate_result": {
            "decision": static_gate.decision,
            "reason": static_gate.reason,
            "generated": static_generated,
            "render_error": static_render_error,
        },
        "sidecar_generation_result": {
            "decision": sidecar_gate.decision,
            "reason": sidecar_gate.reason,
            "sidecars": sidecars,
        },
        "production_ready": "NO",
        "Production Ready": "NO",
        "reason_source_replace_not_used": (
            None if auto_count else "no_high_confidence_auto_source_replace_intervals"
        ),
        "source_replace_not_available_reason": (
            None if auto_count else "balanced_policy_rejected_dynamic_or_uncertain_regions"
        ),
        "expensive_dynamic_replacement_skipped": args.resource_profile != "quality",
        "compute_policy_reason": (
            "Balanced profile disables expensive dynamic plain-text replacement."
            if args.resource_profile == "balanced" else
            "Low profile prefers static fallback."
            if args.resource_profile == "low" else
            "Quality profile still requires confidence before replacement."
        ),
        "decisions": decisions,
        "adaptive_decisions": adaptive_decisions,
    }
    roots = [sidecar_root]
    if includes_replacement:
        roots.append(source_root)
    if includes_static:
        roots.append(static_root)
    for root in roots:
        debug = root / "debug"
        debug.mkdir(parents=True, exist_ok=True)
        (debug / "subtitle_policy_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        decision_sheet = (
            root / "review_frames" / "contact_sheet_policy_decisions.jpg"
        )
        decision_sheet.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (1000, 180), "white")
        ImageDraw.Draw(image).multiline_text(
            (20, 20),
            (
                f"mode={args.subtitle_output_mode} profile={args.resource_profile}\n"
                f"SOURCE_REPLACE={auto_count} STATIC_SUBTITLE_BURN="
                f"{len(fallback_segments) if static_generated else 0} "
                f"REVIEW_REQUIRED={review_count} SKIP={skip_count}\n"
                + (
                    "Source replacement skipped due to safety; "
                    if not auto_count else "Source replacement allowed; "
                )
                + (
                    "static subtitle rendered because target timeline exists."
                    if static_generated else
                    f"static subtitle blocked: {static_gate.reason or 'render_error'}."
                )
            ),
            fill="black", spacing=10,
        )
        image.save(decision_sheet)

    if includes_replacement:
        legacy = {}
        if args.legacy_adaptive_report and args.legacy_adaptive_report.is_file():
            legacy = json.loads(args.legacy_adaptive_report.read_text(encoding="utf-8"))
        legacy_captions = legacy.get("captions", [])
        audit = {
            "sample": args.sample,
            "v5_replaced_count": len(legacy_captions),
            "strict_source_replace_count": auto_count,
            "adaptive_source_replace_count": adaptive_count,
            "strict_policy_gate": {
                "decision": render_policy.replacement_gate(
                    confident_interval_count=auto_count,
                    output_mode_includes_replacement=True,
                ).decision,
                "reason": render_policy.replacement_gate(
                    confident_interval_count=auto_count,
                    output_mode_includes_replacement=True,
                ).reason,
            },
            "adaptive_policy_gate": {
                "decision": replacement_gate.decision,
                "reason": replacement_gate.reason,
            },
            "v5_captions": [
                {
                    "caption_id": caption["caption_id"],
                    "timestamp_range": caption["timestamp_range"],
                    "strategy": caption["strategy"],
                }
                for caption in legacy_captions
            ],
            "adaptive_decisions": adaptive_decisions,
        }
        debug = source_root / "debug"
        (debug / "gate_audit_report.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (debug / "adaptive_source_replace_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
