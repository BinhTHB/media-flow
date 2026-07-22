"""Sweep min_confidence using frozen GT mappings from canonical ASS geometry."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.infrastructure.benchmark.hybrid_subtitle.core import group_frame_candidates, build_tracks
from src.infrastructure.benchmark.hybrid_subtitle.subtitle_classifier import HeuristicSubtitleClassifier

OUT = Path("artifacts/round23")
RAW = OUT / "baseline_raw_yolo_detections.jsonl"
MAPPING = OUT / "gt_track_mapping.json"


def main():
    raw = [json.loads(line) for line in RAW.read_text(encoding="utf-8").splitlines() if line]
    mapping = json.loads(MAPPING.read_text(encoding="utf-8"))
    gt_to_base = {item["gt_id"]: item["base_id"] for item in mapping["matches"]}
    frozen = {item["id"]: item for item in json.loads(Path("scripts/frozen_gt/sample5/frozen_gt.json").read_text(encoding="utf-8"))}
    sub_ids = {track_id for track_id, item in frozen.items() if item["expected_is_subtitle"]}
    rows = []

    for threshold in (0.30, 0.35, 0.40, 0.45, 0.50):
        by_frame = {}
        for detection in raw:
            if float(detection["confidence"]) >= threshold:
                by_frame.setdefault(int(detection["frame_index"]), []).append(detection)
        groups = [
            group
            for frame_index in sorted(by_frame)
            for group in group_frame_candidates(by_frame[frame_index])
        ]
        tracks = build_tracks(groups, iou_threshold=0.25, max_gap_frames=5)
        available_ids = {track["track_id"] for track in tracks}
        survived_subs = sorted(track_id for track_id in sub_ids if gt_to_base[track_id] in available_ids)
        classifier = HeuristicSubtitleClassifier()
        positive = sum(classifier.classify(track, 1280).is_subtitle for track in tracks)
        rows.append({
            "min_confidence": threshold,
            "total_tracks": len(tracks),
            "heuristic_positive_tracks": positive,
            "tracks_would_query_gemini": positive,
            "subtitles_survived": len(survived_subs),
            "subtitles_lost": sorted(sub_ids - set(survived_subs)),
            "gate_pass": len(survived_subs) == 19,
        })

    payload = {
        "frozen_gt_sha256": mapping["frozen_gt_sha256"],
        "mapping_artifact": "artifacts/round23/gt_track_mapping.json",
        "method": "Rebuild tracks from exact round23 raw YOLO detections at every threshold. Verify survival through frozen-GT geometry/time mapping fixed at 0.30.",
        "rows": rows,
    }
    output = OUT / "min_confidence_sweep.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (OUT / "min_confidence_sweep.sha256").write_text(
        hashlib.sha256(output.read_bytes()).hexdigest() + "\n", encoding="ascii"
    )
    print("threshold | tracks | heuristic_pos | GT_sub_survived | gate")
    for row in rows:
        print(f"{row['min_confidence']:.2f} | {row['total_tracks']} | {row['heuristic_positive_tracks']} | {row['subtitles_survived']}/19 | {'PASS' if row['gate_pass'] else 'FAIL'}")

if __name__ == "__main__":
    main()
