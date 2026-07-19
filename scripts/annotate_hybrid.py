from __future__ import annotations
"""
Manual ground-truth annotation tooling for the hybrid subtitle benchmark.

Usage:
  python -c "from scripts.annotate_hybrid import main; main()"
  # Or: python scripts/annotate_hybrid.py sample1 10

This tool generates:
  {sample_id}_manual_gt.json     - Reviewed per-frame annotations
  {sample_id}_subtitle_intervals.json - Timing intervals for subtitle tracks
  {sample_id}_annotation_contact_sheet.png - Visual review sheet
"""
import argparse, base64, cv2, json, os, sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SAMPLES = [
    "test_videos/real_mini_benchmark/sample1.mp4",
    "test_videos/real_mini_benchmark/sample2.mp4",
    "test_videos/real_mini_benchmark/sample3.mp4",
    "test_videos/real_mini_benchmark/sample4.mp4",
    "test_videos/real_mini_benchmark/sample5.mp4",
]


def extract_sample_frames(video_path: str, max_frames: int = 60,
                         interval: int = 1) -> List[Dict[str, Any]]:
    """Extract frames at regular intervals for annotation."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    indices = list(range(0, total, interval))[:max_frames]
    for fi in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            break
        ts = fi / fps * 1000.0
        frames.append({"frame_index": fi, "timestamp_ms": round(ts, 1),
                       "bgr": frame})
    cap.release()
    return frames


def build_annotation_template(video_path: str, output_dir: str,
                               max_frames: int = 60) -> str:
    """Generate annotation JSON template from video frames.
    Annotator reviews each frame and fills in bbox coordinates.
    """
    frames = extract_sample_frames(video_path, max_frames)
    base = Path(video_path).stem
    out_path = os.path.join(output_dir, f"{base}_annotation_template.json")

    # Encode frames as base64 for transportable template
    entries = []
    for f in frames:
        _, buf = cv2.imencode(".jpg", f["bgr"], [cv2.IMWRITE_JPEG_QUALITY, 60])
        entries.append({
            "frame_index": f["frame_index"],
            "timestamp_ms": f["timestamp_ms"],
            "frame_b64": base64.b64encode(buf.tobytes()).decode("ascii"),
            "subtitle_present": None,       # fill: bool
            "subtitle_regions": [],         # fill: [{"bbox_xyxy": [...], "role": "subtitle"}]
            "non_subtitle_text_regions": [],# fill: [{"bbox_xyxy": [...], "role": "..."}]
            "review_status": "pending",
        })

    template = {
        "video_path": video_path,
        "annotator": None,
        "review_status": "pending",
        "annotation_date": None,
        "entries": entries,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)
    return out_path


def build_subtitle_intervals(tracks_path: str, output_dir: str) -> str:
    """Convert text track JSON to human-readable interval list for annotation."""
    import json
    base = Path(tracks_path).stem.replace("_text_tracks", "")
    out_path = os.path.join(output_dir, f"{base}_subtitle_intervals.json")
    try:
        with open(tracks_path, "r", encoding="utf-8") as f:
            tracks = json.load(f)
    except Exception:
        tracks = []

    intervals = []
    for t in tracks:
        intervals.append({
            "track_id": t.get("track_id", ""),
            "start_frame": t.get("start_frame", 0),
            "end_frame": t.get("end_frame", 0),
            "start_ms": t.get("start_ms", 0),
            "end_ms": t.get("end_ms", 0),
            "frame_count": t.get("frame_count", 0),
            "role": "unknown",           # fill: dialogue_subtitle, scene_text, etc.
            "review_status": "pending",
            "confidence": t.get("mean_detector_confidence", 0),
            "safe_cover_bbox": t.get("safe_cover_bbox", []),
            "verdict_color": t.get("verdict_color", "yellow"),
        })

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"sample_id": base, "intervals": intervals,
                   "review_status": "pending"}, f, ensure_ascii=False, indent=2)
    return out_path


def build_contact_sheet(frames: List, labels: List[str],
                         cols: int = 5, scale: float = 0.25) -> Any:
    if not frames:
        return None
    ih, iw = frames[0].shape[:2]
    tw, th = int(iw * scale), int(ih * scale)
    rows = (len(frames) + cols - 1) // cols
    sheet = 255 * __import__("numpy").ones((rows * th, cols * tw, 3), dtype="uint8")
    for i, (fr, lb) in enumerate(zip(frames, labels)):
        if fr is None:
            continue
        thumb = cv2.resize(fr, (tw, th))
        r, c = divmod(i, cols)
        sheet[r*th:(r+1)*th, c*tw:(c+1)*tw] = thumb
        if lb:
            cv2.putText(sheet, lb[:20], (c*tw+2, r*th+14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
    return sheet


def main():
    parser = argparse.ArgumentParser(description="Hybrid benchmark annotation tooling")
    parser.add_argument("--sample", default="sample1")
    parser.add_argument("--output-dir", default=OUT)
    parser.add_argument("--tracks", default=None)  # path to text_tracks.json
    args = parser.parse_args()

    sample = args.sample
    video = f"test_videos/real_mini_benchmark/{sample}.mp4"
    out_dir = args.output_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print(f"[{sample}] Building annotation template from {video}")
    template_path = build_annotation_template(video, out_dir)
    print(f"  Template: {template_path}")

    if args.tracks:
        intervals_path = build_subtitle_intervals(args.tracks, out_dir)
        print(f"  Intervals: {intervals_path}")

    print(f"[{sample}] Annotation tooling complete.")
    print(f"  Review and update {template_path}")
    print(f"  Then rename to {{sample_id}}_manual_gt.json with review_status=reviewed")


if __name__ == "__main__":
    main()
