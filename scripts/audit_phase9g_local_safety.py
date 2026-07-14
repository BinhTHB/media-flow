import json
import subprocess
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
PHASE9F = ROOT / "output" / "phase9f_cover_bbox_alignment_sample1"
OUTPUT = ROOT / "output" / "phase9g_local_safety_audit"
VIDEO = ROOT / "test_videos" / "real_mini_benchmark" / "sample1.mp4"

def _area(box: dict | None) -> float:
    if not box:
        return 0.0
    return max(0.0, box["right"] - box["left"]) * max(0.0, box["bottom"] - box["top"])

def _intersection(left: dict, right: dict) -> float:
    width = max(0.0, min(left["right"], right["right"]) - max(left["left"], right["left"]))
    height = max(0.0, min(left["bottom"], right["bottom"]) - max(left["top"], right["top"]))
    return width * height

def _padded(box: dict | None, width: int, height: int) -> dict | None:
    if not box:
        return None
    x_pad = max(6.0, (box["right"] - box["left"]) * 0.03)
    y_pad = max(6.0, (box["bottom"] - box["top"]) * 0.12)
    return {
        "left": max(0.0, box["left"] - x_pad),
        "top": max(0.0, box["top"] - y_pad),
        "right": min(float(width), box["right"] + x_pad),
        "bottom": min(float(height), box["bottom"] + y_pad),
    }

def _write_sheet(items: list[dict], path: Path) -> None:
    tiles = []
    path.parent.mkdir(parents=True, exist_ok=True)
    for item in items:
        midpoint = sum(item["time_range"]) / 2
        frame = path.parent / f"{item['interval_id']}.jpg"
        subprocess.run(
            ["ffmpeg", "-ss", str(midpoint), "-i", str(VIDEO), "-frames:v", "1",
             "-q:v", "3", "-y", str(frame)],
            check=True, capture_output=True, text=True,
        )
        image = Image.open(frame).convert("RGB")
        draw = ImageDraw.Draw(image)
        source, cover = item["source_subtitle_bbox"], item["cover_box_bbox"]
        if source:
            draw.rectangle(
                (source["left"], source["top"], source["right"], source["bottom"]),
                outline="red", width=8,
            )
        if cover:
            draw.rectangle(
                (cover["left"], cover["top"], cover["right"], cover["bottom"]),
                outline="cyan", width=6,
            )
        text = (
            f"{item['interval_id']} | {item['rejected_by']}\n"
            f"score={item['primary_rejection_score']:.3f} | {item['proposed_action']}"
        )
        draw.rectangle((10, 10, 900, 100), fill="black")
        draw.multiline_text((20, 20), text, fill="white", spacing=6)
        image.thumbnail((360, 640))
        tiles.append(image.copy())
    columns = 4
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (360 * columns, 640 * rows), "black")
    for index, image in enumerate(tiles):
        sheet.paste(image, ((index % columns) * 360, (index // columns) * 640))
    sheet.save(path)

def main() -> None:
    cover = json.loads(
        (PHASE9F / "debug" / "cover_source_subtitle_report.json").read_text(
            encoding="utf-8"
        )
    )
    interval_report = json.loads(
        (PHASE9F / "debug" / "interval_decision_report.json").read_text(
            encoding="utf-8"
        )
    )
    intervals = {item["interval_id"]: item for item in interval_report["intervals"]}
    rejected = [
        item for item in cover["captions"]
        if item.get("skip_reason") == "local_safety_rejected"
    ]
    audited = []
    for caption in rejected:
        interval = intervals[caption["interval_id"]]
        source = caption.get("source_subtitle_bbox")
        evidence = interval.get("person_risk_evidence") or {}
        person_boxes = evidence.get("detected_person_boxes", [])
        source_area = _area(source)
        overlaps = [
            {
                "person_bbox": box,
                "source_overlap_ratio": round(_intersection(source, box) / source_area, 6),
                "person_overlap_ratio": round(_intersection(source, box) / _area(box), 6),
            }
            for box in person_boxes if source and _intersection(source, box)
        ]
        max_overlap = max(
            (item["source_overlap_ratio"] for item in overlaps), default=0.0
        )
        person_score = float(interval.get("person_risk_score", 0.0))
        scene_score = float(interval.get("scene_text_risk_score", 0.0))
        watermark_score = float(interval.get("watermark_risk_score", 0.0))
        if not source:
            classification, action = "correct_reject", "keep_reject"
            rejected_by, visual_label = "missing_source_subtitle_bbox", "no_safe_cover_target"
        elif max_overlap >= 0.35:
            classification, action = "uncertain_review", "manual_review"
            rejected_by, visual_label = "large_person_overlap", "possible_person_overlap"
        else:
            classification, action = "likely_false_reject", "allow_override"
            rejected_by, visual_label = "person_overlap_only", "probable_hog_false_positive"
        cover_box = _padded(source, cover["play_res_x"], cover["play_res_y"])
        audited.append({
            "interval_id": caption["interval_id"],
            "time_range": [
                caption["time_range"]["start"], caption["time_range"]["end"]
            ],
            "source_subtitle_bbox": source,
            "cover_box_bbox": cover_box,
            "rejected_by": rejected_by,
            "primary_rejection_score": max(person_score, scene_score, watermark_score),
            "person_overlap_score": person_score,
            "face_overlap_score": None,
            "scene_text_overlap_score": scene_score,
            "watermark_overlap_score": watermark_score,
            "mask_area_ratio": interval.get("mask_area_ratio"),
            "bbox_area_ratio": (
                round(source_area / (cover["play_res_x"] * cover["play_res_y"]), 6)
                if source else 0.0
            ),
            "target_text_status": caption.get("target_text_mode"),
            "visual_risk_label": visual_label,
            "person_overlap_geometry": overlaps,
            "classification": classification,
            "proposed_action": action,
        })
    classifications = Counter(item["classification"] for item in audited)
    causes = Counter(item["rejected_by"] for item in audited)
    report = {
        "schema_version": "phase9g-local-safety-audit-v1",
        "source_report": str(PHASE9F / "debug" / "cover_source_subtitle_report.json"),
        "summary": {
            "total_rejected": len(audited),
            "correct_reject": classifications["correct_reject"],
            "likely_false_reject": classifications["likely_false_reject"],
            "uncertain_review": classifications["uncertain_review"],
            "top_rejection_causes": dict(causes),
        },
        "intervals": audited,
        "phase9h_proposal": (
            "Allow cover override for person_overlap_only when source bbox exists, "
            "all non-person risks are below 0.20, and source/person overlap is below "
            "0.35; keep missing-bbox rejects and manually review larger overlaps."
        ),
        "production_ready": "NO",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "local_safety_audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_sheet(
        audited,
        OUTPUT / "review_frames" / "contact_sheet_local_safety_rejected.jpg",
    )
    markdown = f"""# Phase 9G Local Safety Audit

## Summary

- Total rejected: {len(audited)}
- Correct reject: {classifications["correct_reject"]}
- Likely false reject: {classifications["likely_false_reject"]}
- Uncertain review: {classifications["uncertain_review"]}
- Top causes: {dict(causes)}

## Finding

The binary HOG person-overlap rule is too strict. It rejects localized bottom subtitle
boxes even when person overlap is small and scene-text and watermark risks are zero.
Missing source bboxes remain correct fail-closed rejections.

## Proposed Phase 9H rule

Allow a local cover override only for `person_overlap_only` when a localized source bbox
exists, all non-person risks are below `0.20`, and source/person overlap is below `0.35`.
Keep missing-bbox rejections and require manual review for larger overlaps.

Production Ready: NO
"""
    (ROOT / "docs" / "phase9g_local_safety_audit.md").write_text(
        markdown, encoding="utf-8"
    )

if __name__ == "__main__":
    main()
