"""Run the reup gate on one finished job.

Exit 0 when source captions are covered, stacked rows stay separate, no
face credit was translated, and the dubbed lines are Vietnamese.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from src.features.visual.reup_gate import (
    box_on_delivery_frame,
    delivery_sample_time,
    dub_coverage,
    gap_row_is_one_panel,
    source_still_visible,
    translated_face_credit,
    translated_production_credit,
)


def _read_frame(path: Path, seconds: float):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, seconds) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return frame


def _crop(frame, box):
    height, width = frame.shape[:2]
    x1 = max(0, min(width - 1, int(box[0]) + 4))
    y1 = max(0, min(height - 1, int(box[1]) + 4))
    x2 = max(x1 + 1, min(width, int(box[2]) - 4))
    y2 = max(y1 + 1, min(height, int(box[3]) - 4))
    return frame[y1:y2, x1:x2]


def _events(job: Path) -> list[dict]:
    payload = json.loads((job / "visual" / "subtitle_events.json").read_text(encoding="utf-8"))
    return list(payload.get("events") or [])


def _box(event: dict) -> list[float]:
    return list(event.get("layout_bbox") or event.get("anchor_bbox"))


def check_visual(job: Path, source: Path, output: Path, variation: dict | None = None) -> list[str]:
    failures: list[str] = []
    profile = variation or {}
    speed = float(profile.get("speed") or 1.0)
    crop_ratio = float(profile.get("crop_ratio") or 0.0)
    mirror = bool(profile.get("mirror"))
    rotate = float(profile.get("rotate_degrees") or 0.0)
    events = [
        event for event in _events(job)
        if event.get("replacement_status") in {"READY", "FRAGMENT", "CREDIT"}
    ]
    if not events:
        return ["no subtitle events"]
    sample = _read_frame(output, delivery_sample_time(float(events[0]["start_time"]), speed))
    if sample is None:
        return ["could not read output frame"]
    frame_h, frame_w = sample.shape[:2]

    def place(box: list[float]) -> list[float]:
        return list(box_on_delivery_frame(box, frame_w, frame_h, crop_ratio, mirror, rotate))

    for event in events:
        box = _box(event)
        placed = place(box)
        if translated_face_credit(placed, frame_w, frame_h, str(event.get("replacement_status"))):
            failures.append(
                f"face credit translated at {float(event['start_time']):.2f}s: {event.get('source_text')}"
            )
        if translated_production_credit(
            str(event.get("replacement_status")),
            str(event.get("source_text") or ""),
        ):
            failures.append(
                f"production credit translated at {float(event['start_time']):.2f}s: {event.get('source_text')}"
            )
        midpoint = (float(event["start_time"]) + float(event["end_time"])) / 2.0
        src = _read_frame(source, midpoint)
        out = _read_frame(output, delivery_sample_time(midpoint, speed))
        if src is None or out is None:
            failures.append(f"missing frame at {midpoint:.2f}s")
            continue
        source_crop = _crop(src, box)
        output_crop = _crop(out, placed)
        if source_crop.size and output_crop.size and source_crop.shape != output_crop.shape:
            output_crop = cv2.resize(
                output_crop,
                (source_crop.shape[1], source_crop.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        if source_still_visible(source_crop, output_crop):
            failures.append(
                f"source caption still visible at {midpoint:.2f}s: {event.get('source_text')}"
            )
    ordered = sorted(events, key=lambda event: (_box(event)[1], float(event["start_time"])))
    for left, right in zip(ordered, ordered[1:]):
        overlap = min(float(left["end_time"]), float(right["end_time"])) - max(
            float(left["start_time"]), float(right["start_time"])
        )
        if overlap < 0.3:
            continue
        upper, lower = (_box(left), _box(right)) if _box(left)[1] <= _box(right)[1] else (_box(right), _box(left))
        midpoint = max(float(left["start_time"]), float(right["start_time"])) + 0.05
        frame = _read_frame(output, delivery_sample_time(midpoint, speed))
        if frame is not None and gap_row_is_one_panel(frame, place(upper), place(lower)):
            failures.append(
                f"stacked rows form one panel at {midpoint:.2f}s: "
                f"{left.get('source_text')} / {right.get('source_text')}"
            )
    return failures


def check_audio(job: Path) -> list[str]:
    path = job / "audio" / "segments.json"
    if not path.is_file():
        return ["no audio/segments.json"]
    segments = json.loads(path.read_text(encoding="utf-8"))
    ok, detail = dub_coverage(segments)
    return [] if ok else [detail]


def audio_matches_source(
    source: Path,
    output: Path,
    start: float,
    variation: dict | None = None,
) -> bool:
    """True when the delivery audio is still the source track.

    A delivery file may already be sped up or pitch-shifted. The source
    snippet is passed through that same change before the comparison, so the
    gate hears the file that will be uploaded.
    """
    profile = variation or {}
    speed = float(profile.get("speed") or 1.0)
    pitch = float(profile.get("pitch_semitones") or 0.0)
    if speed <= 0.0:
        speed = 1.0
    out_start = start / speed
    source_span = 1.0 * speed

    def pcm(cmd: list[str]) -> np.ndarray:
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32)

    output_pcm = pcm([
        "ffmpeg", "-v", "error", "-ss", f"{out_start:.3f}", "-t", "1.0",
        "-i", str(output), "-ac", "1", "-ar", "16000", "-f", "s16le", "-",
    ])
    source_cmd = [
        "ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{source_span:.3f}",
        "-i", str(source),
    ]
    if speed != 1.0 or pitch:
        filters = [f"atempo={speed:.4f}"]
        if pitch:
            ratio = 2.0 ** (pitch / 12.0)
            filters.extend([
                f"asetrate=48000*{ratio:.6f}",
                f"atempo={1.0 / ratio:.6f}",
                "aresample=48000",
            ])
        source_cmd.extend(["-af", ",".join(filters)])
    source_cmd.extend(["-ac", "1", "-ar", "16000", "-f", "s16le", "-"])
    left = pcm(source_cmd)
    right = output_pcm
    count = min(left.size, right.size)
    if count < 8000:
        return False
    left = left[:count] - float(left[:count].mean())
    right = right[:count] - float(right[:count].mean())
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0.0:
        return False
    return float(np.dot(left, right) / denom) > 0.9


def _load_job(job: Path) -> dict:
    path = job / "job.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _dubbing_requested(job: Path) -> bool:
    audio = _load_job(job).get("request", {}).get("audio", {})
    return bool(audio.get("dubbing_enabled", audio.get("enabled", True)))


def _variation(job: Path) -> dict:
    value = _load_job(job).get("variation") or {}
    return value if isinstance(value, dict) else {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check one reup job.")
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    variation = _variation(args.job)
    failures = check_visual(args.job, args.source, args.output, variation)
    if _dubbing_requested(args.job):
        failures.extend(check_audio(args.job))
        segments_path = args.job / "audio" / "segments.json"
        if segments_path.is_file():
            segments = json.loads(segments_path.read_text(encoding="utf-8"))
            dubbed = next((item for item in segments if item.get("tts_provider")), None)
            if dubbed is not None and audio_matches_source(
                args.source, args.output, float(dubbed["start_time"]), variation,
            ):
                failures.append("output audio still matches the source track")
    elif not audio_matches_source(args.source, args.output, 1.0, variation):
        failures.append("final audio is not the source track")
    if failures:
        for item in failures:
            print(f"FAIL {item}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
