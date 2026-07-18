"""Render Sample2 direct-video checkpoint events as localized ASS replacements."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _pixel_box(box: list[float], width: int, height: int) -> dict[str, float]:
    ymin, xmin, ymax, xmax = box
    return {
        "left": xmin * width / 1000,
        "top": ymin * height / 1000,
        "right": xmax * width / 1000,
        "bottom": ymax * height / 1000,
    }


def _ass_time(seconds: float) -> str:
    hours, remainder = divmod(round(seconds * 100), 360000)
    minutes, remainder = divmod(remainder, 6000)
    return f"{hours}:{minutes:02}:{remainder // 100:02}.{remainder % 100:02}"


def _escape_ass(text: str) -> str:
    return text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def build_render_plan(manifest: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    checkpoints = manifest["artifacts"]["checkpoints"]
    accepted = {
        item["candidate_id"]
        for item in checkpoints["classifications"]["payload"]["classifications"]
        if item["classification"] in {"dialogue_subtitle", "narration_subtitle"}
        and item["decision"] == "accepted"
    }
    geometry = {item["event_id"]: item for item in checkpoints["geometry"]["events"]}
    transcriptions = {item["event_id"]: item for item in checkpoints["transcriptions"]["events"]}
    translations = {item["event_id"]: item for item in checkpoints["translations"]["events"]}
    eligible, skipped = [], []
    for event_id, translation in translations.items():
        event = transcriptions.get(event_id)
        geom = geometry.get(event_id)
        if not event or not geom or event["candidate_id"] not in accepted:
            skipped.append({"event_id": event_id, "reason": "missing_prerequisite"})
            continue
        text = str(translation.get("translated_text", "")).strip()
        box = _pixel_box(geom["representative_box_2d"], width, height)
        box_area = (box["right"] - box["left"]) * (box["bottom"] - box["top"])
        if (not text or box["left"] < 0 or box["top"] < 0
                or box["right"] > width or box["bottom"] > height
                or box_area > width * height * 0.45):
            skipped.append({"event_id": event_id, "reason": "invalid_text_or_geometry"})
            continue
        eligible.append({**event, "translated_text": text, "cover_bbox": box})
    return {"eligible": eligible, "skipped": skipped}


def render(plan: dict[str, Any], width: int, height: int, ass_path: Path,
           input_video: Path, output_video: Path) -> None:
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Default,Arial,24,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,5,0,0,0,1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
""".format(width=width, height=height)
    lines = [header]
    for event in plan["eligible"]:
        box = event["cover_bbox"]
        left, top = round(box["left"]), round(box["top"])
        right, bottom = round(box["right"]), round(box["bottom"])
        w, h = right - left, bottom - top
        start, end = _ass_time(event["start_seconds_in_source"]), _ass_time(event["end_seconds_in_source"])
        font_size = max(12, min(24, int(h * 0.55)))
        lines.append(
            f"Dialogue: 0,{start},{end},Default,,0,0,0,,"
            rf"{{\an7\pos({left},{top})\p1\1c&H000000&}}m 0 0 l {w} 0 l {w} {h} l 0 {h}{{\p0}}"
        )
        lines.append(
            f"Dialogue: 1,{start},{end},Default,,0,0,0,,"
            rf"{{\an5\fs{font_size}\pos({round((left + right) / 2)},{round((top + bottom) / 2)})}}"
            + _escape_ass(event["translated_text"])
        )
    ass_path.parent.mkdir(parents=True, exist_ok=True)
    ass_path.write_text("\n".join(lines), encoding="utf-8")
    output_video.parent.mkdir(parents=True, exist_ok=True)
    filter_path = str(ass_path.resolve()).replace("\\", "/").replace(":", r"\:")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(input_video), "-vf", f"ass='{filter_path}'",
         "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-c:a", "copy",
         str(output_video)],
        check=True, capture_output=True, text=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    metadata = manifest["artifacts"]["video_metadata"]
    width, height = int(metadata["width"]), int(metadata["height"])
    plan = build_render_plan(manifest, width, height)
    if not plan["eligible"]:
        raise SystemExit("NO_RENDER_ELIGIBLE_EVENTS")
    ass_path = args.output_root / "subtitles" / "sample2_dynamic_replacement.ass"
    video_path = args.output_root / "videos" / "sample2_dynamic_replacement.mp4"
    render(plan, width, height, ass_path, args.input_video, video_path)
    report = {
        "eligible": len(plan["eligible"]), "skipped": plan["skipped"],
        "ass_path": str(ass_path), "output_path": str(video_path),
        "artifact_hash": hashlib.sha256(ass_path.read_bytes()).hexdigest(),
    }
    report_path = args.output_root / "debug" / "g59_render_plan.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
