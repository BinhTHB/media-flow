"""Parse ASS mask coverage using real geometry mask drawing (\p1 + \pos).

Measures per-frame UNION pixel area from Layer-0 Dialogue events that have:
  - \pos(x,y)       -> top-left corner
  - \p1 + drawing   -> m 0 0 l W 0 l W H l 0 H   (rectangle fill)

Uses ffprobe to get real video frame count/rate/dimensions, so the
parser is portable across resolutions (720p, 1080p, vertical 9:16, etc).

Usage:
  python scripts/parse_mask_coverage.py video.mp4 --ass output.ass
  python scripts/parse_mask_coverage.py video.mp4 --ass output.ass --fps 30 --frames 727
  python scripts/parse_mask_coverage.py --compare video.mp4 baseline.ass new.ass
"""
import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# ffprobe helpers
# ---------------------------------------------------------------------------

def ffprobe_streams(video_path: str) -> Optional[Dict[str, Any]]:
    """Return the first video stream metadata via ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", video_path,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        data = json.loads(r.stdout)
        for s in data.get("streams", []):
            if s.get("codec_type") == "video":
                return s
    except Exception:
        return None
    return None


def get_video_geometry(video_path: str) -> Dict[str, Any]:
    """Return {width, height, fps, frame_count} from ffprobe or fallback."""
    s = ffprobe_streams(video_path)
    if s is None:
        return {"width": 1920, "height": 1080, "fps": 30.0, "frame_count": 0}
    w = int(s.get("width", 1920))
    h = int(s.get("height", 1080))
    # r_frame_rate is "num/den" string
    rfr = s.get("r_frame_rate", "30/1")
    fps = 30.0
    if "/" in rfr:
        try:
            num, den = rfr.split("/")
            fps = float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            fps = 30.0
    # nb_frames is sometimes missing; compute from duration * fps if absent
    nb = s.get("nb_frames")
    if nb is not None:
        try:
            frame_count = int(nb)
        except (ValueError, TypeError):
            frame_count = 0
    else:
        dur_s = _get_duration_seconds(video_path)
        frame_count = int(dur_s * fps) if dur_s else 0
    return {"width": w, "height": h, "fps": fps, "frame_count": frame_count}


def _get_duration_seconds(video_path: str) -> Optional[float]:
    try:
        cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
               "-show_format", video_path]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        data = json.loads(r.stdout)
        dur = data.get("format", {}).get("duration")
        return float(dur) if dur else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# ASS parsing
# ---------------------------------------------------------------------------

_ASS_TIME_RE = re.compile(r"(\d+):(\d+):(\d+)\.(\d+)")


def parse_ass_time(s: str) -> float:
    m = _ASS_TIME_RE.match(s.strip())
    if not m:
        return 0.0
    h, mn, sec, cs = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
    return h * 3600 + mn * 60 + sec + cs / 100.0


def parse_ass_dialogues(ass_path: Path) -> List[Dict[str, Any]]:
    """Extract Layer-0 Dialogue events with geometry mask drawing commands."""
    events: List[Dict[str, Any]] = []
    for line in ass_path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped.startswith("Dialogue: 0,"):
            continue
        # Parse: Dialogue: 0,start,end,style,,0,0,0,,{\an7\pos(x,y)\p1}drawing{\p0}...
        content = stripped[len("Dialogue: 0,"):]
        parts = content.split(",", 8)
        if len(parts) < 9:
            continue
        start_s = parts[0].strip()
        end_s = parts[1].strip()
        text = parts[8]

        # Extract \pos(x,y)
        pos_m = re.search(r"\\pos\((\d+),(\d+)\)", text, re.IGNORECASE)
        if not pos_m:
            continue
        x = int(pos_m.group(1))
        y = int(pos_m.group(2))

        # Extract drawing dimensions m 0 0 l W 0 l W H l 0 H
        draw_m = re.search(
            r"m\s+(\d+)\s+(\d+)\s+l\s+(\d+)\s+(\d+)\s+l\s+(\d+)\s+(\d+)\s+l\s+(\d+)\s+(\d+)",
            text, re.IGNORECASE,
        )
        if not draw_m:
            continue
        # Expected: m 0 0 l W 0 l W H l 0 H
        dw = int(draw_m.group(3))
        dh = int(draw_m.group(6))
        if dw <= 0 or dh <= 0:
            continue

        start_t = parse_ass_time(start_s)
        end_t = parse_ass_time(end_s)
        if end_t <= start_t:
            continue

        events.append({
            "start": start_t,
            "end": end_t,
            "x": x,
            "y": y,
            "width": dw,
            "height": dh,
        })
    return events


# ---------------------------------------------------------------------------
# Coverage computation
# ---------------------------------------------------------------------------

def compute_mask_coverage(
    ass_path: Path,
    geo: Dict[str, Any],
    padding_seconds: float = 0.08,
) -> Dict[str, Any]:
    """Compute frame-wise union coverage from geometry masks."""
    mask_events = parse_ass_dialogues(ass_path)
    W = geo["width"]
    H = geo["height"]
    fps = geo["fps"]
    frame_count = geo["frame_count"]

    if not mask_events or frame_count == 0:
        return {
            "source_ass": str(ass_path),
            "frame_geometry": geo,
            "mask_events_layer_0": len(mask_events),
            "total_frame_count": frame_count,
            "frames_with_active_mask": 0,
            "union_mask_area_pct": {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0},
            "active_mask_boxes_per_frame": {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0},
            "worst_frames": [],
        }

    screen_area = W * H
    frame_areas: List[float] = []
    box_counts: List[int] = []

    for fi in range(frame_count):
        t = fi / fps
        active = [
            e for e in mask_events
            if (e["start"] - padding_seconds) <= t < (e["end"] + padding_seconds)
        ]
        box_counts.append(len(active))
        if not active:
            frame_areas.append(0.0)
            continue

        # Compute union area via pixel raster (numpy not available, use min/max
        # union of axis-aligned rectangles - correct for rectangles which is
        # what mask events always are).
        xs = [max(0, min(W, e["x"])) for e in active]
        ys = [max(0, min(H, e["y"])) for e in active]
        x2s = [max(0, min(W, e["x"] + e["width"])) for e in active]
        y2s = [max(0, min(H, e["y"] + e["height"])) for e in active]

        # Approximation for large sets: segment union on x-projection
        # For accuracy with many overlapping rectangles, we'd need a proper
        # union. But for mask events which are mostly non-overlapping or
        # sparse, this projection-based estimate works well.
        # We use a sweep-line on x to account for horizontal overlap.
        # For each x segment, stack y-intervals.
        total_union = _union_area_rects(list(zip(xs, ys, x2s, y2s)))
        pct = total_union / screen_area * 100.0
        frame_areas.append(pct)

    areas_sorted = sorted(frame_areas)
    n = len(areas_sorted)
    mean = sum(areas_sorted) / n if n else 0.0
    p50 = _percentile(areas_sorted, 50)
    p95 = _percentile(areas_sorted, 95)
    max_area = areas_sorted[-1] if areas_sorted else 0.0
    bc_sorted = sorted(box_counts)
    bc_mean = sum(box_counts) / n if n else 0.0
    bc_median = _percentile([float(b) for b in bc_sorted], 50)
    bc_p95 = _percentile([float(b) for b in bc_sorted], 95)
    bc_max = bc_sorted[-1] if bc_sorted else 0

    frames_active = sum(1 for c in box_counts if c > 0)
    worst_frames = sorted(
        [
            {"frame_index": fi, "active_mask_events": box_counts[fi],
             "union_area_pct": round(frame_areas[fi], 6)}
            for fi in range(n) if frame_areas[fi] > 0
        ],
        key=lambda f: -f["union_area_pct"],
    )[:20]

    return {
        "source_ass": str(ass_path),
        "frame_geometry": geo,
        "mask_events_layer_0": len(mask_events),
        "total_frame_count": frame_count,
        "frames_with_active_mask": frames_active,
        "union_mask_area_pct": {
            "mean": round(mean, 6),
            "p50": round(p50, 6),
            "p95": round(p95, 6),
            "max": round(max_area, 6),
        },
        "active_mask_boxes_per_frame": {
            "mean": round(bc_mean, 4),
            "median": round(bc_median, 4),
            "p95": round(bc_p95, 4),
            "max": int(bc_max),
        },
        "worst_frames": worst_frames,
    }


def _union_area_rects(rects: List[Tuple[int, int, int, int]]) -> int:
    """Union area of axis-aligned rectangles via x-sweep + y-interval union."""
    if not rects:
        return 0

    # Build open/close events
    events: List[Tuple[int, int, int, int]] = []  # (x, open_or_close, y1, y2)
    for x1, y1, x2, y2 in rects:
        if x2 <= x1 or y2 <= y1:
            continue
        events.append((x1, 1, y1, y2))   # open: type=1
        events.append((x2, 0, y1, y2))   # close: type=0

    if not events:
        return 0

    # Sort: x asc, open before close for same x (sweep invariant)
    events.sort(key=lambda e: (e[0], e[1]))

    total = 0
    prev_x = events[0][0]
    # active_y: list of open y-intervals as (y1, y2)
    active_y: List[Tuple[int, int]] = []

    def _y_union_len(ylist: List[Tuple[int, int]]) -> int:
        if not ylist:
            return 0
        merged = sorted(ylist)
        length = 0
        cur_a, cur_b = merged[0]
        for a, b in merged[1:]:
            if a <= cur_b:        # overlapping/adjacent
                cur_b = max(cur_b, b)
            else:                 # disjoint
                length += cur_b - cur_a
                cur_a, cur_b = a, b
        length += cur_b - cur_a
        return length

    for x, kind, y1, y2 in events:
        if x != prev_x:
            total += (x - prev_x) * _y_union_len(active_y)
            prev_x = x
        if kind == 1:   # open
            active_y.append((y1, y2))
        else:           # close
            # Remove one matching interval (first occurrence)
            for i, (a, b) in enumerate(active_y):
                if a == y1 and b == y2:
                    del active_y[i]
                    break

    return total


def _percentile(sorted_list: List[float], q: float) -> float:
    if not sorted_list:
        return 0.0
    import math
    idx = max(0, min(len(sorted_list) - 1, int(math.ceil(q / 100.0 * len(sorted_list)) - 1)))
    return sorted_list[idx]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def compare(baseline_ass: Path, new_ass: Path, video_path: Optional[str] = None) -> None:
    geo = get_video_geometry(video_path) if video_path else {"width": 1920, "height": 1080, "fps": 30.0, "frame_count": 0}
    b = compute_mask_coverage(baseline_ass, geo)
    n = compute_mask_coverage(new_ass, geo)
    bm = b["union_mask_area_pct"]
    nm = n["union_mask_area_pct"]
    print(json.dumps({
        "baseline": b,
        "new": n,
        "delta_mask_count": n["mask_events_layer_0"] - b["mask_events_layer_0"],
        "delta_area_mean": round(nm["mean"] - bm["mean"], 6),
        "delta_area_p95": round(nm["p95"] - bm["p95"], 6),
        "delta_area_max": round(nm["max"] - bm["max"], 6),
    }, indent=2, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse ASS mask coverage using real geometry")
    ap.add_argument("video", nargs="?", help="Video file (for frame geometry)")
    ap.add_argument("--ass", required=False, help="ASS file path")
    ap.add_argument("--fps", type=float, help="Override FPS")
    ap.add_argument("--frames", type=int, help="Override frame count")
    ap.add_argument("--width", type=int, help="Override width")
    ap.add_argument("--height", type=int, help="Override height")
    ap.add_argument("--compare", nargs=2, metavar=("BASELINE", "NEW"), help="Compare two ASS files")
    ap.add_argument("--padding", type=float, default=0.0, help="Optional additional padding seconds")
    ap.add_argument("--output", type=Path, help="Write machine-readable JSON to this path")
    args = ap.parse_args()

    if args.compare:
        compare(Path(args.compare[0]), Path(args.compare[1]), args.video)
        return

    if not args.ass:
        ap.print_help()
        return

    if args.video:
        geo = get_video_geometry(args.video)
    elif None not in (args.width, args.height, args.fps, args.frames):
        geo = {
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "frame_count": args.frames,
        }
    else:
        ap.error("provide video or all of --width, --height, --fps, and --frames")

    if args.fps:
        geo["fps"] = args.fps
    if args.frames is not None:
        geo["frame_count"] = args.frames
    if args.width:
        geo["width"] = args.width
    if args.height:
        geo["height"] = args.height

    result = compute_mask_coverage(Path(args.ass), geo, padding_seconds=args.padding)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    m = result["union_mask_area_pct"]
    b = result["active_mask_boxes_per_frame"]
    print(f"mask={result['mask_events_layer_0']}, "
          f"mean={m['mean']:.4f}%, p95={m['p95']:.4f}%, max={m['max']:.4f}%, "
          f"box_mean={b['mean']:.2f}, box_p95={b['p95']:.2f}")


if __name__ == "__main__":
    main()
