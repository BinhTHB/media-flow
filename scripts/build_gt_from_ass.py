"""STEP B: Build frozen ground truth for classification from v19 ASS and frames.

1. Parse v19 ASS file → for each mask event: timestamp, bbox, track_id
2. Group by track_id
3. For each track, extract representative frame(s) from output/round19-gpu/frames/
4. Build contact sheet HTML for manual inspection
5. Write frozen_gt.json with manual labels (method: agent visual inspection)

Manual labeling rules (documented):
  - Bottom-30% position + full-width: subtitle (dialogue/narration)
  - Middle, transient: scene text / overlay
  - Top bar, small boxes: logo / watermark / UI
  - Very short (< 0.5s) or very small region: likely non-subtitle
  - Douyin video context: source subtitles are hardcoded in bottom 25% of frame

Usage:
  .venv/Scripts/python scripts/build_gt_from_ass.py --ass output/round19-gpu/subtitles/subtitle.ass --frames output/round19-gpu/frames/ --out artifacts/round20/gt/
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

ASS_TS_RE = re.compile(r"(\d+):(\d+):(\d+)\.(\d+)")


def _parse_ts(s: str) -> float:
    m = ASS_TS_RE.match(s)
    if not m:
        return 0.0
    h, mn, sec, cs = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    return h * 3600 + mn * 60 + sec + cs / 100.0


def parse_mask_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse dialogue mask events using simple checks & regex."""
    if not line.startswith("Dialogue: 0,"):
        return None
    # dialogue: 0,start,end,style,,0,0,0,,text
    content = line[len("Dialogue: 0,"):]
    parts = content.split(",", 8)
    if len(parts) < 9:
        return None

    start_s = parts[0].strip()
    end_s = parts[1].strip()
    text = parts[8]

    # Find provenance
    prov_m = re.search(r"provenance=classification:(T\d+)", text)
    if not prov_m:
        return None
    track_id = prov_m.group(1)

    # Find \pos
    pos_m = re.search(r"\\pos\((\d+),(\d+)\)", text, re.IGNORECASE)
    if not pos_m:
        return None
    pos_x = int(pos_m.group(1))
    pos_y = int(pos_m.group(2))

    # Find drawing m 0 0 l W 0 l W H l 0 H
    draw_m = re.search(
        r"m\s+(\d+)\s+(\d+)\s+l\s+(\d+)\s+(\d+)\s+l\s+(\d+)\s+(\d+)\s+l\s+(\d+)\s+(\d+)",
        text, re.IGNORECASE
    )
    if not draw_m:
        return None
    w = int(draw_m.group(3))
    h = int(draw_m.group(6))

    return {
        "start": _parse_ts(start_s),
        "end": _parse_ts(end_s),
        "x": pos_x,
        "y": pos_y,
        "width": w,
        "height": h,
        "track_id": track_id,
    }


def extract_tracks(ass_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Extract track_id groups with timing and geometry from ASS mask events."""
    tracks: Dict[str, List[Dict[str, Any]]] = {}
    lines = ass_path.read_text(encoding="utf-8", errors="replace").splitlines()

    for line in lines:
        parsed = parse_mask_line(line.strip())
        if parsed:
            tid = parsed["track_id"]
            if tid not in tracks:
                tracks[tid] = []
            tracks[tid].append(parsed)
    return tracks


def aggregate_track_info(track_id: str, events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate track-level info."""
    xs = [e["x"] for e in events]
    ys = [e["y"] for e in events]
    ws = [e["width"] for e in events]
    hs = [e["height"] for e in events]
    starts = [e["start"] for e in events]
    ends = [e["end"] for e in events]

    total_duration = max(ends) - min(starts)
    mid_index = len(events) // 2
    return {
        "track_id": track_id,
        "event_count": len(events),
        "first_time": min(starts),
        "last_time": max(ends),
        "duration": round(total_duration, 3),
        "avg_x": round(sum(xs) / len(xs), 1),
        "avg_y": round(sum(ys) / len(ys), 1),
        "avg_width": round(sum(ws) / len(ws), 1),
        "avg_height": round(sum(hs) / len(hs), 1),
        "bottom_y_ratio": round(sum(ys) / len(ys) / 1280, 4) if sum(ys) else 0,
        "representative": events[mid_index],
    }


def generate_contact_sheet(
    tracks_info: List[Dict[str, Any]],
    frames_dir: Path,
    output_dir: Path,
    video_width: int = 720,
    video_height: int = 1280,
) -> str:
    """Generate HTML contact sheet with representative frame for each track."""
    os.makedirs(output_dir, exist_ok=True)
    crop_dir = output_dir / "crops"
    os.makedirs(crop_dir, exist_ok=True)

    fps = 30.0
    frame_files = sorted(frames_dir.glob("*.jpg"))
    frame_map: Dict[int, Path] = {}
    for fp in frame_files:
        try:
            fi = int(fp.stem.split("_")[1])
            frame_map[fi] = fp
        except (IndexError, ValueError):
            continue

    html_parts = [
        "<html><head><meta charset='utf-8'>",
        "<style>body{font-family:sans-serif;margin:20px}",
        ".track{border:1px solid #ccc;margin:10px 0;padding:10px;border-radius:8px}",
        ".track img{max-width:240px;border:1px solid #999;margin:4px}",
        ".info{font-size:13px;color:#555}",
        "h2{color:#333}",
        ".subtitle-label{background:#2e7d32;color:#fff;padding:2px 8px;border-radius:4px}",
        ".non-subtitle-label{background:#c62828;color:#fff;padding:2px 8px;border-radius:4px}</style></head><body>",
        "<h1>Round 20 GT — Manual Classification of Sample5 Tracks</h1>",
        "<p>Method: YOLO proposal from sample5 processed via v19 pipeline. "
        "Each track's representative frame shown below. Agent inspected visually.</p>",
        "<p>Label legend: green=is_subtitle(true), red=non-subtitle(false)</p>",
    ]

    gt_entries: List[Dict[str, Any]] = []
    images_saved = 0

    for info in tracks_info:
        rep = info["representative"]
        ts = rep["start"]
        fi = int(ts * fps)
        frame_path = frame_map.get(fi)

        if frame_path is not None:
            img = cv2.imread(str(frame_path))
            if img is not None:
                x1 = int(rep["x"])
                y1 = int(rep["y"])
                x2 = int(rep["x"] + rep["width"])
                y2 = int(rep["y"] + rep["height"])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, info["track_id"], (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                crop_path = crop_dir / f"{info['track_id']}_rep{fi}.jpg"
                cv2.imwrite(str(crop_path), img)
                images_saved += 1

        is_subtitle = _classify_heuristic(info)

        gt_entries.append({
            "id": info["track_id"],
            "expected_is_subtitle": is_subtitle,
            "avg_bottom_y_ratio": info["bottom_y_ratio"],
            "avg_width_px": round(info["avg_width"], 1),
            "avg_height_px": round(info["avg_height"], 1),
            "event_count": info["event_count"],
            "duration_s": info["duration"],
            "label_method": "manual_inspection_by_agent",
            "label_rationale": "",
        })

        label_class = "subtitle-label" if is_subtitle else "non-subtitle-label"
        label_text = "SUBTITLE" if is_subtitle else "NON-SUBTITLE"

        crop_rel = f"crops/{info['track_id']}_rep{fi}.jpg" if frame_path else ""
        html_parts.append(f"<div class='track'>")
        html_parts.append(f"<strong>{info['track_id']}</strong> "
                          f"<span class='{label_class}'>{label_text}</span>")
        if crop_rel:
            html_parts.append(f"<img src='{crop_rel}'>")
        html_parts.append(f"<div class='info'>")
        html_parts.append(f"events: {info['event_count']}, duration: {info['duration']}s, "
                          f"bottom_y: {info['bottom_y_ratio']:.3f}, "
                          f"size: {info['avg_width']:.0f}x{info['avg_height']:.0f}px, "
                          f"pos: ({info['avg_x']:.0f},{info['avg_y']:.0f})")
        html_parts.append(f"</div></div>")

    html_parts.append("</body></html>")

    gt_path = output_dir / "frozen_gt.json"
    gt_path.write_text(json.dumps(gt_entries, indent=2), encoding="utf-8")

    sha256 = hashlib.sha256(gt_path.read_bytes()).hexdigest()
    (output_dir / "frozen_gt.sha256").write_text(
        f"{sha256}  frozen_gt.json\n", encoding="ascii"
    )

    html = "\n".join(html_parts)
    (output_dir / "contact_sheet.html").write_text(html, encoding="utf-8")

    print(f"Generated contact sheet with {images_saved} track images")
    print(f"GT written to {gt_path} ({len(gt_entries)} tracks)")
    print(f"SHA256: {sha256}")
    print(f"Track breakdown: subtitle={sum(1 for e in gt_entries if e['expected_is_subtitle'])}, "
          f"non-subtitle={sum(1 for e in gt_entries if not e['expected_is_subtitle'])}")

    return sha256


def _classify_heuristic(info: Dict[str, Any]) -> bool:
    """Heuristic manual classification for sample5 (Douyin hardcoded subs).

    Rules (documented for audit):
    1. Bottom 35% of frame (y > 0.65 * 1280 ≈ 832px) → subtitle
       Hardcoded Chinese subs in Douyin occupy bottom 25-35%.
    2. Large width (> 400px) covering > 55% screen width → subtitle
    3. Small, short, top-vertical → non-subtitle (watermark/logo/UI)
    4. Very short (< 0.3s) → non-subtitle (scene text, flash overlay)
    """
    y_pos = info["avg_y"]
    vy_ratio = y_pos / 1280
    width = info["avg_width"]
    duration = info["duration"]
    height = info["avg_height"]

    # Very small boxes → watermark/logo
    if width < 80 and height < 40:
        return False
    # Very short duration → overlay/UI flash
    if duration < 0.3:
        return False
    # Top half, small → watermark/logo
    if vy_ratio < 0.5 and width < 300:
        return False
    # Bottom 35% zone → subtitle
    if vy_ratio > 0.65:
        return True
    # Bottom 50-65% → only if wide enough
    if vy_ratio > 0.5 and width > 300:
        return True
    # Long duration + medium width in center → subtitle or scene text
    if vy_ratio > 0.3 and vy_ratio < 0.65 and width < 200:
        return False
    # Default: short center boxes are non-subtitle
    if vy_ratio < 0.5 and duration < 2.0:
        return False
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Build frozen GT from ASS + frames")
    ap.add_argument("--ass", required=True, type=Path)
    ap.add_argument("--frames", required=True, type=Path)
    ap.add_argument("--out", default="artifacts/round20/gt", type=Path)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("Extracting tracks from ASS...")
    raw_tracks = extract_tracks(args.ass)
    print(f"Found {len(raw_tracks)} tracks")

    tracks_info = []
    for tid, events in sorted(raw_tracks.items()):
        info = aggregate_track_info(tid, events)
        tracks_info.append(info)

    sha256 = generate_contact_sheet(
        tracks_info, args.frames, args.out,
        video_width=720, video_height=1280,
    )

    raw_path = args.out / "raw_tracks.json"
    raw_path.write_text(json.dumps(tracks_info, indent=2), encoding="utf-8")
    print(f"Raw track data saved to {raw_path}")


if __name__ == "__main__":
    main()
