"""Extract start/end timestamps from the round19 ASS file and write sidecar JSON."""
import json
import re
from pathlib import Path

ass_path = Path("output/round19-gpu/subtitles/subtitle.ass")
gt_path = Path("scripts/frozen_gt/sample5/frozen_gt.json")
out_dir = Path("artifacts/round23")

def _parse_ts(s: str) -> float:
    m = re.match(r"(\d+):(\d+):(\d+)\.(\d+)", s)
    if not m:
        return 0.0
    h, mn, sec, cs = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    return h * 3600 + mn * 60 + sec + cs / 100.0

def main():
    lines = ass_path.read_text(encoding="utf-8", errors="replace").splitlines()
    
    # Map track_id -> list of event dicts
    track_events = {}
    
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("Dialogue: 0,"):
            continue
        content = stripped[len("Dialogue: 0,"):]
        parts = content.split(",", 8)
        if len(parts) < 9:
            continue
        start_s = parts[0].strip()
        end_s = parts[1].strip()
        text = parts[8]
        
        # Extract provenance
        prov_m = re.search(r"provenance=classification:(T\d+)", text)
        if not prov_m:
            continue
        tid = prov_m.group(1)
        
        start_t = _parse_ts(start_s)
        end_t = _parse_ts(end_s)
        
        track_events.setdefault(tid, []).append((start_t, end_t))
        
    # Aggregate sidecar timestamps
    sidecar = {}
    for tid, events in track_events.items():
        starts = [e[0] for e in events]
        ends = [e[1] for e in events]
        sidecar[tid] = {
            "start_time_s": min(starts),
            "end_time_s": max(ends),
            "duration_s": max(ends) - min(starts),
        }
        
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "gt_track_timestamps.json", "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)
        
    print(f"Extracted sidecar timestamps for {len(sidecar)} tracks.")
    # Check T003
    print("T003 timestamps:", sidecar.get("T003"))

if __name__ == "__main__":
    main()
