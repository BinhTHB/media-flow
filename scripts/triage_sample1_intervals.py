"""Create a review package for mixed subtitle/text intervals in sample1."""

import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    root = Path("output/real_video_benchmark_results/sample1")
    report = json.loads((root / "debug/interval_eligibility_report_asr.json").read_text(encoding="utf-8"))
    tracks = defaultdict(list)
    triage = []
    buckets = defaultdict(list)
    for item in report["intervals"]:
        candidate = item.get("selected_candidate")
        action = (
            "SAFE_AUTO_REPLACE_CANDIDATE" if item["decision"] == "AUTO_REPLACE"
            else "SKIP" if item["decision"] == "SKIP_REPLACEMENT"
            else "UNSUPPORTED_SCENE_TEXT" if item["scene_text_risk_score"] >= 0.5
            else "MANUAL_REGION_NEEDED"
        )
        bbox = item.get("source_bbox") or (candidate or {}).get("bbox")
        if bbox:
            tracks[round((bbox["top"] + bbox["bottom"]) / 2 / 100) * 100].append(item)
        record = {
            "interval_id": item["interval_id"], "timestamp_range": {"start": item["timestamp_start"], "end": item["timestamp_end"]},
            "current_decision": item["decision"], "selected_candidate": candidate,
            "ocr_text": (candidate or {}).get("raw_ocr_text"), "asr_segment_text": item.get("asr_segment_text"),
            "asr_ocr_score": item.get("asr_ocr_consistency_score"), "scene_text_risk_score": item["scene_text_risk_score"],
            "subtitle_likeness_score": item["subtitle_likeness_score"], "source_subtitle_present_score": item["source_subtitle_present_score"],
            "mask_area_ratio": item["mask_area_ratio"], "reason": item.get("skip_reason") or item.get("review_reason"),
            "recommended_action": action,
        }
        triage.append(record)
        buckets[action].append(record)
    hypotheses = [{
        "track_id": f"track_{index}", "y_range": [key - 50, key + 50],
        "temporal_density": round(len(values) / len(triage), 4),
        "asr_ocr_match_rate": round(sum(bool(value.get("asr_ocr_consistency_score")) for value in values) / len(values), 4),
        "scene_text_risk": round(sum(value["scene_text_risk_score"] for value in values) / len(values), 4),
        "subtitle_likelihood": round(sum(value["subtitle_likeness_score"] for value in values) / len(values), 4),
        "recommended_classification": "unknown_track",
    } for index, (key, values) in enumerate(sorted(tracks.items()))]
    output = {"intervals": triage, "safe_auto_replace_candidates": buckets["SAFE_AUTO_REPLACE_CANDIDATE"], "review_candidates": buckets["MANUAL_REGION_NEEDED"], "skip_candidates": buckets["SKIP"], "track_hypotheses": hypotheses}
    (root / "debug/sample1_interval_triage_report.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "manual_interval_review_template.json").write_text(json.dumps([
        {"interval_id": item["interval_id"], "timestamp_start": item["timestamp_range"]["start"], "timestamp_end": item["timestamp_range"]["end"], "current_decision": item["current_decision"], "recommended_action": item["recommended_action"], "user_decision": None, "manual_source_bbox": None, "manual_target_bbox": None, "notes": ""}
        for item in triage
    ], ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "sample1_interval_triage_report.md").write_text(
        "\n".join(["# Sample1 interval triage", "", f"- Safe candidates: {len(buckets['SAFE_AUTO_REPLACE_CANDIDATE'])}", f"- Manual review: {len(buckets['MANUAL_REGION_NEEDED'])}", f"- Skip: {len(buckets['SKIP'])}"]), encoding="utf-8"
    )
    frames = root / "frames"
    review = root / "review_frames"
    for name, items in {"safe_candidates": buckets["SAFE_AUTO_REPLACE_CANDIDATE"], "review_required_detailed": buckets["MANUAL_REGION_NEEDED"], "skipped_intervals": buckets["SKIP"], "asr_ocr_mismatch": [item for item in triage if item["asr_ocr_score"] == 0.0], "scene_text_rejections": [item for item in triage if item["scene_text_risk_score"] > 0]}.items():
        images = []
        for item in items:
            path = frames / f"frame_{item['timestamp_range']['start']:g}s.jpg"
            if path.exists():
                image = Image.open(path).convert("RGB")
                ImageDraw.Draw(image).text((15, 15), f"{item['current_decision']}: {item['reason']}", fill="red")
                images.append(image)
        if images:
            sheet = Image.new("RGB", (images[0].width, images[0].height * len(images)))
            for index, image in enumerate(images): sheet.paste(image, (0, index * image.height))
            sheet.save(review / f"contact_sheet_{name}.jpg")


if __name__ == "__main__":
    main()
