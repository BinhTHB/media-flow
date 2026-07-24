"""Parse contact_sheet.html safely block-by-block."""
import json
import re
from pathlib import Path

html_path = Path("scripts/frozen_gt/sample5/contact_sheet.html")
html = html_path.read_text(encoding="utf-8")

# Split into track blocks
blocks = html.split("<div class='track'>")
# The first block is header, ignore
blocks = blocks[1:]

tracks = []
for idx, block in enumerate(blocks):
    # Extract track_id
    id_match = re.search(r"<strong>(T\d+)</strong>", block)
    if not id_match:
        continue
    tid = id_match.group(1)

    # Extract start_frame
    frame_match = re.search(r"crops/T\d+_rep(\d+)\.jpg", block)
    if not frame_match:
        continue
    start_frame = int(frame_match.group(1))

    # Extract info fields: events: 16, duration: 0.7s, bottom_y: 0.747, size: 134x76px, pos: (292,956)
    info_match = re.search(
        r"events:\s+(\d+),\s+duration:\s+([\d.]+)s,\s+bottom_y:\s+([\d.]+),\s+size:\s+(\d+)x(\d+)px,\s+pos:\s+\((\d+),(\d+)\)",
        block,
    )
    if not info_match:
        print(f"WARN: info match failed for {tid}")
        continue

    duration_s = float(info_match.group(2))
    bottom_y_ratio = float(info_match.group(3))
    width_px = float(info_match.group(4))
    height_px = float(info_match.group(5))
    pos_x = float(info_match.group(6))
    pos_y = float(info_match.group(7))  # ASS pos is bottom-left

    y2 = pos_y
    y1 = y2 - height_px
    x1 = pos_x
    x2 = x1 + width_px

    tracks.append({
        "track_id": tid,
        "start_time": start_frame / 30.0,
        "start_frame": start_frame,
        "duration_s": duration_s,
        "bbox_xyxy": [x1, y1, x2, y2],
        "avg_bottom_y_ratio": bottom_y_ratio,
        "avg_width_px": width_px,
        "avg_height_px": height_px,
    })

print(f"Parsed {len(tracks)} tracks successfully.")
# Save
output_dir = Path("artifacts/round23")
output_dir.mkdir(parents=True, exist_ok=True)
with open(output_dir / "gt_tracks_with_geometry.json", "w", encoding="utf-8") as f:
    json.dump(tracks, f, indent=2)
print("Saved to artifacts/round23/gt_tracks_with_geometry.json")

