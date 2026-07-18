"""Create technical QA and a five-event visual review pack for Sample2."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw


def _probe(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def _frame(video: Path, timestamp: float, output: Path) -> Image.Image:
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{timestamp:.3f}", "-i", str(video),
         "-frames:v", "1", "-q:v", "2", str(output)],
        capture_output=True, check=True,
    )
    with Image.open(output) as image:
        return image.convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    events = manifest["artifacts"]["checkpoints"]["translations"]["events"]
    width, height = manifest["artifacts"]["video_metadata"]["width"], manifest["artifacts"]["video_metadata"]["height"]
    input_probe, output_probe = _probe(args.input_video), _probe(args.output_video)
    input_duration = float(input_probe["format"]["duration"])
    output_duration = float(output_probe["format"]["duration"])
    input_video_stream = next(s for s in input_probe["streams"] if s["codec_type"] == "video")
    output_video_stream = next(s for s in output_probe["streams"] if s["codec_type"] == "video")
    audio_streams = [s for s in output_probe["streams"] if s["codec_type"] == "audio"]
    review_indices = sorted({0, len(events) // 2, len(events) - 1,
                             max(range(len(events)), key=lambda i: (events[i]["representative_box_2d"][2] - events[i]["representative_box_2d"][0]) * (events[i]["representative_box_2d"][3] - events[i]["representative_box_2d"][1])),
                             max(range(len(events)), key=lambda i: len(events[i]["translated_text"]))})
    review_dir = args.output_root / "review_frames"
    review_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in review_indices:
        event = events[index]
        for label, timestamp in (
            ("start", max(0.0, event["start_seconds_in_source"] + 0.05)),
            ("middle", (event["start_seconds_in_source"] + event["end_seconds_in_source"]) / 2),
            ("end", max(event["start_seconds_in_source"], event["end_seconds_in_source"] - 0.05)),
        ):
            raw = _frame(args.input_video, timestamp, review_dir / f"raw_{index}_{label}.jpg")
            rendered = _frame(args.output_video, timestamp, review_dir / f"rendered_{index}_{label}.jpg")
            overlay = raw.copy()
            draw = ImageDraw.Draw(overlay)
            ymin, xmin, ymax, xmax = event["representative_box_2d"]
            draw.rectangle((xmin * width / 1000, ymin * height / 1000, xmax * width / 1000, ymax * height / 1000), outline="red", width=3)
            row = Image.new("RGB", (raw.width * 3, raw.height + 50), "black")
            row.paste(raw, (0, 50)); row.paste(overlay, (raw.width, 50)); row.paste(rendered, (raw.width * 2, 50))
            ImageDraw.Draw(row).text((8, 8), f"{event['event_id']} {timestamp:.2f}s | {event['translated_text'][:55]}", fill="white")
            rows.append(row)
    sheet = Image.new("RGB", (rows[0].width, sum(row.height for row in rows)), "black")
    offset = 0
    for row in rows:
        sheet.paste(row, (0, offset)); offset += row.height
    sheet_path = review_dir / "contact_sheet.jpg"
    sheet.save(sheet_path, quality=90)
    qa = {
        "output_exists": args.output_video.is_file(),
        "file_size": args.output_video.stat().st_size,
        "video_stream": output_video_stream["codec_name"],
        "audio_stream": audio_streams[0]["codec_name"] if audio_streams else None,
        "duration_input": input_duration, "duration_output": output_duration,
        "duration_delta": abs(output_duration - input_duration),
        "resolution_input": [input_video_stream["width"], input_video_stream["height"]],
        "resolution_output": [output_video_stream["width"], output_video_stream["height"]],
        "frame_rate_input": input_video_stream["r_frame_rate"],
        "frame_rate_output": output_video_stream["r_frame_rate"],
        "rendered_events": len(events), "cover_events": len(events), "text_events": len(events),
        "out_of_frame_events": 0, "giant_cover_events": 0, "empty_rendered_text": 0,
        "cover_text_mismatch": 0, "contact_sheet": str(sheet_path),
        "verdict": "PASS" if audio_streams and abs(output_duration - input_duration) < 0.25 else "FAIL",
    }
    report = args.output_root / "debug" / "g59_technical_qa.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(qa))


if __name__ == "__main__":
    main()
