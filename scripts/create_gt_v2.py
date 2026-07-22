"""Create immutable GT V2 from existing independently specified mechanics fixtures."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path("test_videos/subtitle_replacement_benchmark")
OUT = Path("artifacts/round23")
SETS = ("moving", "center")


def main() -> None:
    entries = []
    for set_name in SETS:
        payload = json.loads((ROOT / set_name / "ground_truth.json").read_text(encoding="utf-8"))
        for timestamp in payload["checkpoints"]:
            key = str(int(timestamp)) if float(timestamp).is_integer() else str(timestamp)
            bbox = payload["expected_source_subtitle_bbox_per_timestamp"][key]
            entries.append({
                "id": f"{set_name}:{key}",
                "set": set_name,
                "timestamp_s": timestamp,
                "is_subtitle": True,
                "bbox_xyxy": [bbox["left"], bbox["top"], bbox["right"], bbox["bottom"]],
            })
    result = {
        "schema_version": 1,
        "source": "existing subtitle_replacement_benchmark ground_truth.json fixtures",
        "entries": entries,
    }
    target = OUT / "gt_v2.json"
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    (OUT / "gt_v2.sha256").write_text(f"{digest}  gt_v2.json\n", encoding="ascii")
    print(f"entries={len(entries)} sha256={digest}")

if __name__ == "__main__":
    main()
