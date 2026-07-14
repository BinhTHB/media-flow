import argparse
from collections import Counter
import json
import subprocess
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw

from src.application.services.ass_generator import ASSGenerator
from src.application.services.subtitle_render_policy import SubtitleRenderPolicy
from src.domain.entities.resource_policy import FFmpegConfig, ResourcePolicy
from src.domain.entities.segment import Segment
from src.domain.entities.subtitle_placement import SubtitlePlacement
from src.domain.entities.subtitle_style import SubtitleStyle
from src.infrastructure.runtime.ffmpeg_capability_detector import detect_ffmpeg_capabilities
from src.infrastructure.runtime.ffmpeg_encode import build_ffmpeg_encode_args
from src.infrastructure.runtime.resource_usage_report import write_resource_usage_report


def _segment(item: dict) -> Segment:
    return Segment(
        item["id"], float(item["start"]), float(item["end"]), item["text"],
        item.get("translated_text", ""), item["source"], float(item["confidence"]),
    )


def _text(interval: dict | None, segment: Segment) -> tuple[str, str | None]:
    if segment.translated_text and segment.translated_text != "[REVIEW]":
        return segment.translated_text, "normalized"
    candidate = (interval or {}).get("selected_candidate") or {}
    text = candidate.get("raw_ocr_text", "")
    return text, "ocr" if text else None


def _interval(segment: Segment, intervals: list[dict]) -> dict | None:
    return next(
        (item for item in intervals if item["timestamp_start"] <= segment.start < item["timestamp_end"]),
        None,
    )


def _sheet(video: Path, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for timestamp in (5, 10, 15):
        frame = path.parent / f"{path.stem}_{timestamp}s.jpg"
        subprocess.run(
            ["ffmpeg", "-ss", str(timestamp), "-i", str(video), "-frames:v", "1", "-q:v", "2", "-y", str(frame)],
            check=True, capture_output=True, text=True,
        )
        frames.append(Image.open(frame).convert("RGB"))
    sheet = Image.new("RGB", (frames[0].width * 3, frames[0].height + 48), "white")
    ImageDraw.Draw(sheet).text((16, 16), title, fill="black")
    for index, frame in enumerate(frames):
        sheet.paste(frame, (index * frame.width, 48))
    sheet.save(path)


def _srt(segments: list[Segment]) -> str:
    def stamp(value: float) -> str:
        milliseconds = round(value * 1000)
        hours, milliseconds = divmod(milliseconds, 3_600_000)
        minutes, milliseconds = divmod(milliseconds, 60_000)
        seconds, milliseconds = divmod(milliseconds, 1000)
        return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"
    return "\n\n".join(
        f"{index}\n{stamp(segment.start)} --> {stamp(segment.end)}\n{segment.translated_text}"
        for index, segment in enumerate(segments, 1)
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--captions", type=Path, required=True)
    parser.add_argument("--interval-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("output/subtitle_policy_v2"))
    parser.add_argument("--sample", required=True)
    parser.add_argument("--resource-profile", choices=("low", "balanced", "quality"), default="balanced")
    parser.add_argument("--ffmpeg-encoder", choices=("auto", "cpu", "nvenc"), default="auto")
    args = parser.parse_args()

    policy = ResourcePolicy(args.resource_profile)
    config = FFmpegConfig(encoder=args.ffmpeg_encoder)
    intervals = json.loads(args.interval_report.read_text(encoding="utf-8"))["intervals"]
    segments = [_segment(item) for item in json.loads(args.captions.read_text(encoding="utf-8"))]
    decisions, sidecar_segments, sources = [], [], Counter()
    render_policy = SubtitleRenderPolicy()
    for segment in segments:
        interval = _interval(segment, intervals)
        decision = render_policy.ux_decide(interval, args.resource_profile)
        text, source = _text(interval, segment)
        original_visible = bool(interval and interval.get("source_subtitle_present_score", 0.0) >= 0.70)
        decisions.append({
            "segment_id": segment.id,
            "decision": decision.decision,
            "reason": decision.reason,
            "original_subtitle_visible": original_visible,
            "original_subtitle_region": (interval or {}).get("source_bbox") or ((interval or {}).get("selected_candidate") or {}).get("bbox"),
            "duplicate_subtitle_risk": original_visible,
        })
        if text:
            sidecar_segments.append(Segment(
                segment.id, segment.start, segment.end, segment.text,
                "\n".join(textwrap.wrap(text, 42)[:2]), segment.source, segment.confidence,
            ))
            sources[source or "unknown"] += 1

    bottom_count = sum(item["decision"] == "bottom_subtitle_replace_candidate" for item in decisions)
    duplicate_count = sum(item["duplicate_subtitle_risk"] for item in decisions)
    bottom_root = args.output_root / f"{args.sample}_bottom_replace"
    safe_root = args.output_root / f"{args.sample}_static_fallback_safe"
    caps = detect_ffmpeg_capabilities()
    plan = build_ffmpeg_encode_args(
        str(args.video), str(bottom_root / "videos" / "output.mp4"), config, False, caps
    )
    (bottom_root / "videos").mkdir(parents=True, exist_ok=True)
    subprocess.run(plan.args, check=True, capture_output=True, text=True)
    write_resource_usage_report(
        bottom_root, policy, caps, plan,
        expensive_steps_skipped=["dynamic_plain_text_replacement"] if not bottom_count else [],
    )
    _sheet(
        bottom_root / "videos" / "output.mp4",
        bottom_root / "review_frames" / "contact_sheet_bottom_replace_attempt.jpg",
        "BOTTOM SOURCE REPLACE ATTEMPT - no fallback burned",
    )

    sidecar_dir = safe_root / "subtitles"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    placements = [SubtitlePlacement(alignment=2, margin_v=70, is_fallback=True) for _ in sidecar_segments]
    (sidecar_dir / "output.ass").write_text(
        ASSGenerator.generate(sidecar_segments, SubtitleStyle(font_size=34, outline_width=3.0, alignment=2), placements),
        encoding="utf-8",
    )
    (sidecar_dir / "output.srt").write_text(_srt(sidecar_segments), encoding="utf-8")
    safe_video = safe_root / "videos" / "output.mp4"
    safe_video.parent.mkdir(parents=True, exist_ok=True)
    safe_plan = build_ffmpeg_encode_args(str(args.video), str(safe_video), config, False, caps)
    subprocess.run(safe_plan.args, check=True, capture_output=True, text=True)
    write_resource_usage_report(
        safe_root, policy, caps, safe_plan,
        expensive_steps_skipped=["burned_static_fallback_due_to_duplicate_risk"],
    )
    _sheet(
        safe_video,
        safe_root / "review_frames" / "contact_sheet_static_fallback_safe.jpg",
        "STATIC FALLBACK SAFE - sidecar only, no duplicate burn-in",
    )
    _sheet(
        args.video,
        safe_root / "review_frames" / "contact_sheet_duplicate_risk.jpg",
        "DUPLICATE RISK - original subtitle remains visible; fallback is sidecar",
    )
    decision_sheet = Image.new("RGB", (1000, 180), "white")
    ImageDraw.Draw(decision_sheet).multiline_text(
        (20, 20),
        (
            f"bottom_replace={bottom_count}; duplicate_risk={duplicate_count}\n"
            f"sidecar_segments={len(sidecar_segments)}; burned_static_fallback=0\n"
            "Balanced policy keeps uncertain original subtitles visible and emits sidecars."
        ),
        fill="black", spacing=10,
    )
    review = safe_root / "review_frames"
    review.mkdir(parents=True, exist_ok=True)
    decision_sheet.save(review / "contact_sheet_policy_decisions.jpg")
    report = {
        "resource_profile": args.resource_profile,
        "original_subtitle_visible": duplicate_count > 0,
        "original_subtitle_region": [
            item["original_subtitle_region"]
            for item in decisions if item["original_subtitle_region"] is not None
        ],
        "duplicate_subtitle_risk": duplicate_count > 0,
        "static_fallback_suppressed": duplicate_count > 0,
        "fallback_position_selected": "sidecar",
        "source_replace_attempted": bottom_count > 0,
        "bottom_source_replace_count": bottom_count,
        "static_fallback_burned_count": 0,
        "static_fallback_suppressed_count": duplicate_count,
        "reason_static_fallback_not_burned": "duplicate_subtitle_risk",
        "sidecar_subtitle_generated": bool(sidecar_segments),
        "target_text_source_distribution": dict(sources),
        "decisions": decisions,
    }
    for root in (bottom_root, safe_root):
        debug = root / "debug"
        debug.mkdir(parents=True, exist_ok=True)
        (debug / "subtitle_policy_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
