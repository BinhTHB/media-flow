import os
import subprocess
import argparse
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "test_videos" / "real_mini_benchmark"

def process_sample(sample: str, input_root: Path, output_root: Path) -> None:
    video_path = INPUT_DIR / f"{sample}.mp4"
    sample_output = input_root / sample
    qa_sample_dir = output_root / sample
    qa_review_dir = qa_sample_dir / "review_frames"
    qa_review_dir.mkdir(parents=True, exist_ok=True)

    cover_review_frames = list((sample_output / "_static_cover" / "review_frames").glob("frame_*s.jpg"))
    if not cover_review_frames:
        print(f"No review frames found for {sample}")
        return

    timestamps = []
    for f in cover_review_frames:
        t_str = f.name.replace("frame_", "").replace("s.jpg", "")
        timestamps.append(float(t_str))
    timestamps.sort()
    overlay_frames = list(
        (sample_output / "debug" / "adaptive_overlays").glob("frame_*s.jpg")
    )

    def nearest_overlay(timestamp: float) -> Path | None:
        if not overlay_frames:
            return None
        return min(
            overlay_frames,
            key=lambda path: abs(
                float(path.name.replace("frame_", "").replace("s.jpg", ""))
                - timestamp
            ),
        )

    rows = []
    for t in timestamps:
        t_str = f"{t:g}s"
        raw_frame_path = qa_review_dir / f"temp_raw_{t_str}.jpg"
        subprocess.run([
            "ffmpeg", "-ss", str(t), "-i", str(video_path),
            "-frames:v", "1", "-q:v", "2", "-y", str(raw_frame_path)
        ], capture_output=True, check=True)

        cover_path = sample_output / "_static_cover" / "review_frames" / f"frame_{t_str}.jpg"
        burn_path = sample_output / "_static_burn" / "review_frames" / f"frame_{t_str}.jpg"
        det_path = nearest_overlay(t)

        img_raw = Image.open(raw_frame_path).convert("RGB")
        img_cover = Image.open(cover_path).convert("RGB")
        img_burn = Image.open(burn_path).convert("RGB")
        
        if det_path is not None:
            img_det = Image.open(det_path).convert("RGB")
        else:
            img_det = Image.new("RGB", img_raw.size, "black")

        w, h = img_raw.size
        row = Image.new("RGB", (w * 4, h))
        row.paste(img_raw, (0, 0))
        row.paste(img_cover, (w, 0))
        row.paste(img_burn, (w * 2, 0))
        row.paste(img_det, (w * 3, 0))

        draw = ImageDraw.Draw(row)
        draw.rectangle((0, 0, w * 4, 40), fill="black")
        draw.text((10, 10), f"TIMESTAMP: {t_str} | INPUT FRAME", fill="white")
        draw.text((w + 10, 10), "STATIC SOURCE COVER", fill="white")
        draw.text((w * 2 + 10, 10), "STATIC SUBTITLE BURN", fill="white")
        draw.text((w * 3 + 10, 10), "REGION DETECTION OVERLAY", fill="white")

        rows.append(row)
        if raw_frame_path.exists():
            os.remove(raw_frame_path)

    if rows:
        w_total = rows[0].width
        h_total = sum(r.height for r in rows)
        contact_sheet = Image.new("RGB", (w_total, h_total))
        y_offset = 0
        for r in rows:
            contact_sheet.paste(r, (0, y_offset))
            y_offset += r.height
        
        contact_sheet_path = qa_review_dir / "contact_sheet_input_vs_cover_vs_fallback.jpg"
        contact_sheet.save(contact_sheet_path, quality=90)
        print(f"Saved contact sheet for {sample} to {contact_sheet_path}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root", type=Path,
        default=ROOT / "output" / "static_source_cover_mvp",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=ROOT / "output" / "static_source_cover_mvp_qa",
    )
    args = parser.parse_args()
    for sample in ("sample2", "sample4", "sample5"):
        process_sample(sample, args.input_root, args.output_root)

if __name__ == "__main__":
    main()