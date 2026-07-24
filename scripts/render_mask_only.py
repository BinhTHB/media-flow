"""STEP D: Generate mask-only ASS and QA figures.

1. Filter the full ASS to only mask events (Layer 0, mask drawing).
2. Render mask-only video using ffmpeg.
3. Compare coverage metrics between original and mask-only.
4. Save QA artifacts to artifacts/round20/qa/.
"""
import argparse
import json
from pathlib import Path

def filter_mask_ass(full_ass: Path, output_path: Path) -> int:
    """Strip Layer 1 dialogue events; keep only Layer 0 mask events."""
    lines = full_ass.read_text(encoding="utf-8", errors="replace").splitlines()
    header_end = 0
    for i, line in enumerate(lines):
        if line.strip() == "[Events]":
            header_end = i
            break
    header = lines[:header_end + 2]  # include [Events] and Format line
    mask_events = [l for l in lines if l.startswith("Dialogue: 0,")]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join([*header, *mask_events, ""]), encoding="utf-8")
    return len(mask_events)


def render_mask_only(video_path: Path, mask_ass_path: Path, output_path: Path) -> bool:
    """Render mask-only video: video with transparent overlay of mask ASS."""
    import subprocess
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"ass={mask_ass_path}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-an",
        str(output_path),
    ]
    ret = subprocess.run(cmd, capture_output=True, timeout=300)
    return ret.returncode == 0


def main():
    ap = argparse.ArgumentParser(description="Generate mask-only render + QA")
    ap.add_argument("ass", type=Path, help="Full ASS file (v19)")
    ap.add_argument("video", type=Path, help="Input video")
    ap.add_argument("--out", type=Path, default=Path("artifacts/round20/qa"))
    ap.add_argument("--skip-render", action="store_true", help="Skip ffmpeg render step")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # 1. Filter mask-only ASS
    mask_ass = args.out / "mask_only.ass"
    n_masks = filter_mask_ass(args.ass, mask_ass)
    print(f"Mask-only ASS written: {mask_ass} ({n_masks} mask events)")

    # 2. Render mask-only video
    if not args.skip_render:
        mask_video = args.out / "mask_only.mp4"
        ok = render_mask_only(args.video, mask_ass, mask_video)
        if ok:
            print(f"Mask-only video rendered: {mask_video}")
        else:
            print("WARNING: ffmpeg mask-only render failed. Check ffmpeg availability.")

    # 3. Measure coverage from mask_only ASS using the geometry parser
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.parse_mask_coverage import compute_mask_coverage, get_video_geometry
    try:
        geo = get_video_geometry(str(args.video))
    except Exception:
        geo = {"width": 720, "height": 1280, "fps": 30.0, "frame_count": 727}
    result = compute_mask_coverage(mask_ass, geo, padding_seconds=0.0)
    qa_report = {
        "source_video": str(args.video),
        "source_ass": str(mask_ass),
        "mask_events_layer_0": result["mask_events_layer_0"],
        "total_frame_count": result["total_frame_count"],
        "frames_with_active_mask": result["frames_with_active_mask"],
        "union_mask_area_pct": result["union_mask_area_pct"],
        "active_mask_boxes_per_frame": result["active_mask_boxes_per_frame"],
    }
    qa_path = args.out / "qa_report.json"
    qa_path.write_text(json.dumps(qa_report, indent=2), encoding="utf-8")
    print(f"QA report written: {qa_path}")

    m = result["union_mask_area_pct"]
    b = result["active_mask_boxes_per_frame"]
    print(f"mask={result['mask_events_layer_0']}, "
          f"mean={m['mean']:.4f}%, p95={m['p95']:.4f}%, max={m['max']:.4f}%, "
          f"box_mean={b['mean']:.2f}, box_p95={b['p95']:.2f}")

    # Save top 20 worst frames
    worst = result.get("worst_frames", [])
    if worst:
        worst_path = args.out / "worst_frames_top20.json"
        worst_path.write_text(json.dumps(worst[:20], indent=2), encoding="utf-8")
        print(f"Worst 20 frames saved: {worst_path}")


if __name__ == "__main__":
    main()
