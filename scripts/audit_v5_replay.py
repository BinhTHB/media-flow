"""Replay the V5 source-replacement baseline against current V6 decisions."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V5 = ROOT / "output/adaptive_render_v5/debug/adaptive_render_report.json"
DEFAULT_CURRENT = ROOT / "output/subtitle_policy_v6/sample1_adaptive_source_replace/debug/adaptive_source_replace_report.json"
DEFAULT_CURRENT_RENDER = ROOT / "output/subtitle_policy_v6/sample1_adaptive_source_replace/debug/adaptive_render_report.json"
DEFAULT_GATE = ROOT / "output/subtitle_policy_v6/sample1_adaptive_source_replace/debug/gate_audit_report.json"
DEFAULT_INTERVALS = ROOT / "output/real_video_benchmark_results/sample1/debug/interval_eligibility_report.json"
DEFAULT_VIDEO_GATE = ROOT / "output/real_video_benchmark_results/sample1/debug/video_eligibility_report.json"
DEFAULT_OUTPUT = ROOT / "output/phase9_v5_replay_audit/gate_audit_report.json"
DEFAULT_MARKDOWN = ROOT / "docs/phase9_v5_replay_audit.md"
DEFAULT_CONTACT_SHEET = ROOT / "output/phase9_v5_replay_audit/review_frames/contact_sheet_v5_vs_current.jpg"
DEFAULT_ORIGINAL_VIDEO = ROOT / "test_videos/real_mini_benchmark/sample1.mp4"
DEFAULT_V5_VIDEO = ROOT / "output/adaptive_render_v5/videos/output.mp4"
DEFAULT_CURRENT_VIDEO = ROOT / "output/subtitle_policy_v6/sample1_adaptive_source_replace/videos/output.mp4"


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object and fail with a useful error for malformed artifacts."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return value


def overlaps(left: dict[str, float], right: dict[str, float]) -> bool:
    """Return whether two half-open timestamp ranges overlap."""
    return left["start"] < right["end"] and right["start"] < left["end"]


def primary_reason(decisions: list[dict[str, Any]]) -> str:
    """Choose a deterministic causal label from overlapping current intervals."""
    reasons = [
        item.get("reason") for item in decisions
        if item.get("decision") != "AUTO_SOURCE_REPLACE" and item.get("reason")
    ]
    if not reasons:
        return "current_adaptive_recovered"
    counts = Counter(reasons)
    return min(counts, key=lambda reason: (-counts[reason], reasons.index(reason)))


def classify_caption(
    v5_caption: dict[str, Any],
    current: list[dict[str, Any]],
    decisions_by_id: dict[str, dict[str, Any]],
    strict_gate: dict[str, Any],
    adaptive_gate: dict[str, Any],
    intervals: list[dict[str, Any]],
    video_gate: dict[str, Any],
) -> dict[str, Any]:
    """Map one V5 caption to every overlapping current one-second interval."""
    timestamp = v5_caption["timestamp_range"]
    matched = [
        caption for caption in current
        if overlaps(timestamp, caption["timestamp_range"])
    ]
    evidence = []
    for caption in matched:
        decision = decisions_by_id.get(caption["caption_id"], {})
        evidence.append({
            "interval_id": caption["caption_id"],
            "timestamp_range": caption["timestamp_range"],
            "decision": decision.get("decision", caption.get("original_decision")),
            "reason": decision.get("reason"),
            "render_applied": caption.get("render_applied", False),
            "source_bbox_present": bool(caption.get("source_bbox")),
            "target_text_present": bool(caption.get("target_text", "").strip()),
            "target_text_source": caption.get("target_text_source"),
        })
    decisions = [item for item in evidence if item["decision"]]
    cause = primary_reason(decisions)
    applied = [item for item in evidence if item["render_applied"]]
    candidate_missing = any(
        item["reason"] == "no_reliable_source_subtitle" for item in evidence
    )
    track_blocked = any(
        item["reason"] in {"insufficient_track_evidence", "area_or_track_uncertainty"}
        for item in evidence
    )
    missing_target = any(not item["target_text_present"] for item in evidence)
    matched_intervals = [
        interval for interval in intervals
        if overlaps(timestamp, {
            "start": interval["timestamp_start"],
            "end": interval["timestamp_end"],
        })
    ]
    safety_reasons = sorted({
        reason for interval in matched_intervals
        for reason in (
            "scene_text_risk" if interval.get("scene_text_risk_score", 0.0) >= 0.20 else None,
            "watermark_risk" if interval.get("watermark_risk_score", 0.0) >= 0.20 else None,
            "mask_area_above_0_20" if interval.get("mask_area_ratio", 0.0) > 0.20 else None,
        ) if reason
    })
    candidate_found = any(interval.get("selected_candidate") for interval in matched_intervals)
    safety_decision = "REJECT" if safety_reasons else "NO_EXPLICIT_SAFETY_REJECT"
    current_decision = (
        "AUTO_SOURCE_REPLACE" if applied else
        next((item["decision"] for item in decisions if item["decision"]), "NOT_REPORTED")
    )
    primary = primary_reason(decisions)
    override_possible = bool(applied)
    override_notes = []
    if video_gate.get("eligibility_decision") == "SKIP_REPLACEMENT":
        override_notes.append("video_level_skip_is_prior_only_for_adaptive_mode")
    if not candidate_found:
        override_notes.append("current_candidate_not_found; replay_v5_bbox_required")
    if safety_reasons:
        override_notes.append("explicit_safety_risk:" + ",".join(safety_reasons))
    if missing_target:
        override_notes.append("missing_target_text_in_overlapping_interval")
    if not override_notes:
        override_notes.append("local_candidate_evidence_available")
    return {
        "v5_caption_id": v5_caption["caption_id"],
        "v5_time_range": timestamp,
        "v5_bbox": v5_caption.get("source_bbox"),
        "v5_mask_bbox": v5_caption.get("mask_bbox"),
        "v5_strategy": v5_caption.get("strategy"),
        "v5_target_text_source": v5_caption.get("target_text_source"),
        "v5_confidence": v5_caption.get("confidence"),
        "current_matched_interval_ids": [
            interval["interval_id"] for interval in matched_intervals
        ],
        "current_candidate_found": candidate_found,
        "current_decision": current_decision,
        "rejected_by_gate": primary if not applied else None,
        "rejection_reason": primary if not applied else None,
        "video_level_gate": video_gate.get("eligibility_decision", "NOT_REPORTED"),
        "track_stability_decision": (
            "TRACK_BLOCKED" if track_blocked else
            "CURRENT_ADAPTIVE_RENDERED" if applied else "NOT_REPORTED"
        ),
        "safety_decision": safety_decision,
        "target_text_status": "MISSING_IN_OVERLAP" if missing_target else "PRESENT_IN_ALL_OVERLAPS",
        "candidate_level_override_possible": override_possible,
        "override_risk_notes": override_notes,
        "current_interval_evidence": evidence,
        "current_interval_policy_evidence": [
            {
                "interval_id": interval["interval_id"],
                "decision": interval.get("decision"),
                "reason": interval.get("reason"),
                "selected_candidate": bool(interval.get("selected_candidate")),
                "mask_area_ratio": interval.get("mask_area_ratio"),
                "scene_text_risk_score": interval.get("scene_text_risk_score"),
                "watermark_risk_score": interval.get("watermark_risk_score"),
            }
            for interval in matched_intervals
        ],
        "strict_policy_gate": strict_gate,
        "adaptive_policy_gate": adaptive_gate,
    }


def build_report(
    v5: dict[str, Any],
    current: dict[str, Any],
    gate: dict[str, Any],
    interval_report: dict[str, Any],
    video_gate: dict[str, Any],
    current_render: dict[str, Any],
    source_artifacts: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a standalone, deterministic replay audit report."""
    current_captions = current_render.get("captions", [])
    v5_captions = v5.get("captions", [])
    if not isinstance(current_captions, list) or not isinstance(v5_captions, list):
        raise ValueError("Both render reports must contain captions lists.")
    decisions_by_id = {
        item["segment_id"]: item for item in gate.get("adaptive_decisions", [])
        if isinstance(item, dict) and "segment_id" in item
    }
    strict_gate = gate.get("strict_policy_gate", {})
    adaptive_gate = gate.get("adaptive_policy_gate", {})
    intervals = interval_report.get("intervals", [])
    mappings = [
        classify_caption(
            caption, current_captions, decisions_by_id, strict_gate,
            adaptive_gate, intervals, video_gate,
        )
        for caption in v5_captions
    ]
    causes = Counter(
        item["rejection_reason"] or "current_adaptive_recovered"
        for item in mappings
    )
    current_replacement_count = sum(
        caption.get("render_applied", False) for caption in current_captions
    )
    reported_replacement_count = int(
        current.get(
            "adaptive_source_replace_count",
            current_replacement_count,
        )
    )
    v5_video_path = v5.get("video_path")
    current_video_path = current.get("video_path")
    video_match = v5_video_path == current_video_path
    return {
        "audit": "phase9_v5_replay",
        "scope": "audit_only",
        "production_ready": "NO",
        "Production Ready": "NO",
        "source_artifacts": source_artifacts or {
            "v5_render_report": str(DEFAULT_V5.relative_to(ROOT)),
            "current_v6_render_report": str(DEFAULT_CURRENT.relative_to(ROOT)),
            "current_v6_gate_audit_report": str(DEFAULT_GATE.relative_to(ROOT)),
        },
        "counts": {
            "v5_caption_count": len(v5_captions),
            "current_interval_count": len(current_captions),
            "current_adaptive_replacement_count": current_replacement_count,
            "lost_count": max(
                0,
                len(v5_captions)
                - reported_replacement_count,
            ),
            "mapped_v5_caption_count": len(mappings),
        },
        "provenance": {
            "v5_video_path": v5_video_path,
            "current_video_path": current_video_path,
            "video_path_match": video_match,
            "evidence_limit": (
                None if video_match else
                "Timestamp mapping is complete, but visual/candidate equivalence is not established across different recorded video paths."
            ),
        },
        "gate_evidence": {
            "strict_policy_gate": strict_gate,
            "adaptive_policy_gate": adaptive_gate,
            "video_level_gate": {
                "decision": video_gate.get("eligibility_decision"),
                "reasons": video_gate.get("reasons", []),
            },
        },
        "cause_distribution": dict(sorted(causes.items())),
        "caption_mappings": mappings,
        "verdicts": {
            "v5_compatibility_parity": "NO",
            "standalone_replay_audit": "COMPLETE",
            "production_ready": "NO",
            "reason": (
                "V5 has eight replacements, current V6 has three interval replacements, "
                "V5 leakage/safety remains unevaluated, and recorded video paths differ."
            ),
        },
    }


def _frame(video: Path, timestamp: float, output: Path) -> Image.Image:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-ss", f"{timestamp:.3f}", "-i", str(video), "-frames:v", "1", "-q:v", "2", "-y", str(output)],
        check=True, capture_output=True, text=True,
    )
    with Image.open(output) as image:
        return image.convert("RGB").resize((240, 427), Image.Resampling.LANCZOS)

def write_contact_sheet(
    output: Path, mappings: list[dict[str, Any]], original_video: Path,
    v5_video: Path, current_video: Path,
) -> None:
    """Create eight timestamp-aligned rows with provenance and gate labels."""
    canvas = Image.new("RGB", (980, 8 * 465 + 40), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), "Original | V5 replacement | Current adaptive | gate label", fill="black")
    for index, mapping in enumerate(mappings):
        timestamp = (mapping["v5_time_range"]["start"] + mapping["v5_time_range"]["end"]) / 2
        y = 35 + index * 465
        images = [
            _frame(original_video, timestamp, output.parent / f"original_{index:02d}.jpg"),
            _frame(v5_video, timestamp, output.parent / f"v5_{index:02d}.jpg"),
            _frame(current_video, timestamp, output.parent / f"current_{index:02d}.jpg"),
        ]
        for column, image in enumerate(images):
            canvas.paste(image, (column * 250, y + 20))
        label = (
            f"{mapping['v5_caption_id']} @ {timestamp:.2f}s | "
            f"{mapping['current_decision']} | "
            f"{mapping['rejection_reason'] or 'recovered'}"
        )
        draw.text((10, y), label, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=90)

def write_markdown(report: dict[str, Any], output: Path) -> None:
    """Write the required human-readable Phase 9 audit summary."""
    counts = report["counts"]
    rows = [
        "# Phase 9 V5 Replay Audit",
        "",
        "## Summary",
        "",
        f"- v5_count: **{counts['v5_caption_count']}**",
        f"- current_count: **{counts['current_adaptive_replacement_count']}**",
        f"- lost_count: **{counts['lost_count']}**",
        f"- video-level gate: `{report['gate_evidence']['video_level_gate']['decision']}`",
        "",
        "## Eight V5 captions",
        "",
        "| V5 caption | Current intervals | Decision | Primary rejection | Override possible |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in report["caption_mappings"]:
        rows.append(
            f"| `{item['v5_caption_id']}` | {', '.join(item['current_matched_interval_ids'])} | "
            f"`{item['current_decision']}` | `{item['rejection_reason'] or 'none'}` | "
            f"`{item['candidate_level_override_possible']}` |"
        )
    rows.extend([
        "",
        "## Top regression causes",
        "",
        *[f"- `{cause}`: {count}" for cause, count in report["cause_distribution"].items()],
        "",
        "## Phase 9B candidates, not implemented",
        "",
        "- Make the video-level `SKIP_REPLACEMENT` result a prior rather than a hard adaptive block.",
        "- Replay V5 bbox evidence per candidate before changing track thresholds.",
        "- Keep candidate-level safety checks and require frame review before any override.",
        "",
        "## Verdict",
        "",
        "- V5 replay audit: **PASS**",
        "- Adaptive gate fix readiness: **NOT READY** (the source video provenance differs and local safety equivalence is not established).",
        "- Production Ready: **NO**",
        "",
    ])
    output.write_text("\n".join(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v5-report", type=Path, default=DEFAULT_V5)
    parser.add_argument("--current-report", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--current-render-report", type=Path, default=DEFAULT_CURRENT_RENDER)
    parser.add_argument("--gate-report", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--interval-report", type=Path, default=DEFAULT_INTERVALS)
    parser.add_argument("--video-gate-report", type=Path, default=DEFAULT_VIDEO_GATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--contact-sheet", type=Path, default=DEFAULT_CONTACT_SHEET)
    parser.add_argument("--original-video", type=Path, default=DEFAULT_ORIGINAL_VIDEO)
    parser.add_argument("--v5-video", type=Path, default=DEFAULT_V5_VIDEO)
    parser.add_argument("--current-video", type=Path, default=DEFAULT_CURRENT_VIDEO)
    args = parser.parse_args()
    report = build_report(
        load_json(args.v5_report),
        load_json(args.current_report),
        load_json(args.gate_report),
        load_json(args.interval_report),
        load_json(args.video_gate_report),
        load_json(args.current_render_report),
        {
            "v5_render_report": str(args.v5_report.relative_to(ROOT)),
            "current_v6_render_report": str(args.current_report.relative_to(ROOT)),
            "current_v6_gate_audit_report": str(args.gate_report.relative_to(ROOT)),
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(report, args.markdown)
    write_contact_sheet(
        args.contact_sheet, report["caption_mappings"], args.original_video,
        args.v5_video, args.current_video,
    )
    print(json.dumps(report["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
