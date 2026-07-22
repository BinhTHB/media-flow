"""Generate exact single-decode GPU YOLO baseline for round23 mapping."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.infrastructure.benchmark.hybrid_subtitle.yolo_proposal import YoloTextProposalProvider
from src.infrastructure.benchmark.hybrid_subtitle.core import group_frame_candidates, build_tracks

OUT = Path("artifacts/round23")
VIDEO = "input/sample5.mp4"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    provider = YoloTextProposalProvider(
        weights_path="best.pt", conf=0.05, iou=0.70, imgsz=1024, device="cuda"
    )
    bundle = provider.detect_video(VIDEO, stride=1)
    raw_detections = bundle.detections
    (OUT / "baseline_proposal_bundle_metadata.json").write_text(
        json.dumps(bundle.to_metadata(), indent=2), encoding="utf-8"
    )
    with open(OUT / "baseline_raw_yolo_detections.jsonl", "w", encoding="utf-8") as f:
        for detection in raw_detections:
            f.write(json.dumps(detection) + "\n")

    for threshold in (0.30,):
        by_frame = {}
        for detection in raw_detections:
            if float(detection["confidence"]) >= threshold:
                by_frame.setdefault(int(detection["frame_index"]), []).append(detection)
        groups = [
            group
            for frame_index in sorted(by_frame)
            for group in group_frame_candidates(by_frame[frame_index])
        ]
        tracks = build_tracks(groups, iou_threshold=0.25, max_gap_frames=5)
        track_path = OUT / f"baseline_tracks_conf_{threshold:.2f}.json"
        track_path.write_text(json.dumps(tracks, indent=2), encoding="utf-8")
        sha = hashlib.sha256(track_path.read_bytes()).hexdigest()
        (OUT / f"baseline_tracks_conf_{threshold:.2f}.sha256").write_text(
            f"{sha}  {track_path.name}\n", encoding="ascii"
        )
        print(f"threshold={threshold:.2f} tracks={len(tracks)} sha={sha}")


if __name__ == "__main__":
    main()
