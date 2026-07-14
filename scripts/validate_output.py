"""Developer validation tool for output job artifacts."""

import argparse
import json
import subprocess
from pathlib import Path


ARTIFACTS = (
    "audio/audio.wav",
    "transcripts/transcript.json",
    "transcripts/translated.json",
    "transcripts/normalized.json",
    "transcripts/segmented.json",
    "subtitles/subtitle.srt",
    "subtitles/subtitle.ass",
    "videos/output.mp4",
)


def load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def duration(segment: dict) -> float:
    return max(segment["end"] - segment["start"], 0.0)


def cjk_count(text: str) -> int:
    return sum("\u4e00" <= char <= "\u9fff" for char in text)


def timing_metrics(segments: list[dict]) -> dict:
    durations = [duration(segment) for segment in segments]
    gaps, overlaps = [], []
    for previous, current in zip(segments, segments[1:]):
        delta = current["start"] - previous["end"]
        if delta > 0.1:
            gaps.append(delta)
        elif delta < -0.1:
            overlaps.append(-delta)
    return {
        "count": len(segments),
        "duration_average": round(sum(durations) / len(durations), 3) if durations else 0,
        "duration_shortest": round(min(durations), 3) if durations else 0,
        "duration_longest": round(max(durations), 3) if durations else 0,
        "duration_under_0_8s": sum(value < 0.8 for value in durations),
        "duration_over_6s": sum(value > 6.0 for value in durations),
        "gap_count": len(gaps),
        "gap_total": round(sum(gaps), 3),
        "gap_longest": round(max(gaps), 3) if gaps else 0,
        "overlap_count": len(overlaps),
    }


def text_metrics(segments: list[dict], key: str) -> dict:
    values = [segment.get(key, "").strip() for segment in segments]
    return {
        "empty_count": sum(not value for value in values),
        "adjacent_duplicate_count": sum(
            current == previous for previous, current in zip(values, values[1:])
        ),
        "cjk_leftover_count": sum(cjk_count(value) > 0 for value in values),
    }


def srt_time(value: str) -> float:
    hours, minutes, seconds = value.replace(",", ".").split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def srt_entries(path: Path) -> list[dict]:
    entries = []
    for block in path.read_text(encoding="utf-8").strip().split("\n\n"):
        lines = block.splitlines()
        if len(lines) < 3 or " --> " not in lines[1]:
            continue
        start, end = lines[1].split(" --> ")
        entries.append(
            {
                "start": srt_time(start),
                "end": srt_time(end),
                "text": "\n".join(lines[2:]),
            }
        )
    return entries


def subtitle_metrics(job_dir: Path) -> dict:
    entries = srt_entries(job_dir / "subtitles" / "subtitle.srt")
    metrics = timing_metrics(entries)
    lines = [line for entry in entries for line in entry["text"].splitlines()]
    metrics.update(
        {
            "max_chars_per_line": max(map(len, lines), default=0),
            "max_lines_per_subtitle": max(
                (len(entry["text"].splitlines()) for entry in entries), default=0
            ),
            "duplicate_text_count": len(entries)
            - len({entry["text"] for entry in entries}),
        }
    )
    return metrics


def normalized_text_preserved(normalized: list[dict], segmented: list[dict]) -> bool:
    """Check that each normalized translation is recoverable from its child segments."""
    children: dict[str, list[str]] = {}
    for segment in segmented:
        parent_id = segment["id"].rsplit("_", 1)[0]
        children.setdefault(parent_id, []).append(segment.get("translated_text", ""))

    for segment in normalized:
        expected = " ".join(segment.get("translated_text", "").replace("\n", " ").split())
        actual = " ".join(
            " ".join(child.replace("\n", " ").split())
            for child in children.get(segment["id"], [segment.get("translated_text", "")])
        )
        if expected != actual:
            return False
    return True


def ffprobe(video_path: Path) -> dict:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=15)
    payload = json.loads(result.stdout)
    streams = payload["streams"]
    video_streams = [stream for stream in streams if stream["codec_type"] == "video"]
    audio_streams = [stream for stream in streams if stream["codec_type"] == "audio"]
    return {
        "file_size_bytes": video_path.stat().st_size,
        "duration": round(float(payload["format"]["duration"]), 3),
        "video_stream_count": len(video_streams),
        "audio_stream_count": len(audio_streams),
        "video_codec": video_streams[0]["codec_name"] if video_streams else None,
        "width": video_streams[0].get("width") if video_streams else None,
        "height": video_streams[0].get("height") if video_streams else None,
    }


def extract_frames(job_dir: Path, video_path: Path, video_duration: float) -> list[str]:
    frame_dir = job_dir / "validation" / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for label, position in (("early", 0.15), ("middle", 0.50), ("late", 0.85)):
        path = frame_dir / f"{label}.jpg"
        command = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{video_duration * position:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "3",
            str(path),
        ]
        subprocess.run(command, capture_output=True, check=True, timeout=30)
        outputs.append(str(path.relative_to(job_dir)))
    return outputs


def build_report(job_dir: Path, extract_sample_frames: bool = False) -> dict:
    report = {
        "job_dir": str(job_dir),
        "artifacts": {
            "missing": [item for item in ARTIFACTS if not (job_dir / item).is_file()],
        },
    }
    report["artifacts"]["present_count"] = len(ARTIFACTS) - len(report["artifacts"]["missing"])

    transcript = load_json(job_dir / "transcripts" / "transcript.json")
    translated = load_json(job_dir / "transcripts" / "translated.json")
    normalized = load_json(job_dir / "transcripts" / "normalized.json")
    segmented = load_json(job_dir / "transcripts" / "segmented.json")
    report["transcript"] = timing_metrics(transcript) | text_metrics(transcript, "text")
    report["translation"] = timing_metrics(translated) | text_metrics(translated, "translated_text")
    report["translation"]["untranslated_count"] = sum(
        not segment.get("translated_text", "").strip()
        or segment["translated_text"].strip() == segment.get("text", "").strip()
        for segment in translated
    )
    report["segmented"] = timing_metrics(segmented) | text_metrics(segmented, "translated_text")
    report["subtitle"] = subtitle_metrics(job_dir)
    report["count_checks"] = {
        "segmented_count": len(segmented),
        "srt_count": len(srt_entries(job_dir / "subtitles" / "subtitle.srt")),
        "ass_dialogue_count": sum(
            line.startswith("Dialogue:")
            for line in (job_dir / "subtitles" / "subtitle.ass").read_text(encoding="utf-8").splitlines()
        ),
        "normalized_text_preserved_in_segmented": normalized_text_preserved(
            normalized, segmented
        ),
    }
    video_path = job_dir / "videos" / "output.mp4"
    report["render"] = ffprobe(video_path)
    if extract_sample_frames:
        report["sample_frames"] = extract_frames(
            job_dir, video_path, report["render"]["duration"]
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True, type=Path)
    parser.add_argument("--frames", action="store_true", help="Extract three sample frames.")
    args = parser.parse_args()
    if not args.job_dir.is_dir():
        parser.error(f"job directory does not exist: {args.job_dir}")
    print(json.dumps(build_report(args.job_dir, args.frames), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
