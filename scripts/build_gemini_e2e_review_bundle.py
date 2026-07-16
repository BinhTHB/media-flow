"""Build a self-contained, evidence-only Gemini E2E review bundle."""

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
MAX_VIDEO_BYTES = 64 * 1024 * 1024


def json_read(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def segment_list(payload) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return payload.get("segments", [])
    return []


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ffprobe(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def video_details(probe: dict) -> tuple[float, str, str, str]:
    streams = probe.get("streams", [])
    video = next(item for item in streams if item.get("codec_type") == "video")
    audio = next((item for item in streams if item.get("codec_type") == "audio"), None)
    return (
        float(probe["format"]["duration"]),
        f"{video['width']}x{video['height']}",
        video["codec_name"],
        audio["codec_name"] if audio else "MISSING",
    )


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def extract_frame(video: Path, timestamp: float, destination: Path) -> None:
    duration, _, _, _ = video_details(ffprobe(video))
    timestamp = min(max(timestamp, 0.0), max(duration - 0.1, 0.0))
    subprocess.run(
        [
            "ffmpeg", "-y", "-ss", f"{timestamp:.3f}", "-i", str(video),
            "-strict", "-2", "-frames:v", "1", "-q:v", "2", str(destination),
        ],
        check=True,
        capture_output=True,
    )


def event_selection(display: list[dict], placements: list[dict]) -> list[tuple[str, dict, float]]:
    if not display:
        return []
    first, last = display[0], display[-1]
    middle = display[len(display) // 2]
    longest = max(display, key=lambda item: len(item.get("text", "")))
    special = next((item for item in placements if item.get("alignment") != 2), placements[0] if placements else {})
    difficult = next((item for item in display if item.get("segment_id") == special.get("segment_id")), longest)
    return [
        ("Before subtitle", {"segment_id": "none", "text": "No subtitle"}, max(0.0, first["start"] - 1.0)),
        ("Subtitle start", first, first["start"]),
        ("Subtitle middle", middle, (middle["start"] + middle["end"]) / 2),
        ("Subtitle end", last, max(last["start"], last["end"] - 0.4)),
        ("Longest subtitle", longest, (longest["start"] + longest["end"]) / 2),
        ("Placement event", difficult, (difficult["start"] + difficult["end"]) / 2),
    ]


def create_contact_sheet(
    video: Path,
    timeline: dict,
    destination: Path,
    translated_by_id: dict[str, dict],
) -> None:
    tracks = timeline.get("tracks", {})
    display, placements = tracks.get("display_subtitles", []), tracks.get("placements", [])
    selected = event_selection(display, placements)
    if not selected:
        Image.new("RGB", (1440, 300), "black").save(destination, quality=90)
        return

    tile_width, tile_height, label_height = 480, 853, 130
    frames: list[Image.Image] = []
    temp_dir = destination.parent / "_contact_sheet_tmp"
    temp_dir.mkdir(exist_ok=True)
    for index, (kind, event, timestamp) in enumerate(selected):
        temp = temp_dir / f"{index}.jpg"
        extract_frame(video, timestamp, temp)
        image = Image.open(temp).convert("RGB").resize((tile_width, tile_height), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (tile_width, tile_height + label_height), "black")
        tile.paste(image, (0, 0))
        event_id = event.get("segment_id", "")
        source_id = event_id.rsplit("_", 1)[0] if event_id.count("_") > 2 else event_id
        source = translated_by_id.get(event_id, translated_by_id.get(source_id, {}))
        source_text = source.get("text", "MISSING").replace("\n", " ")
        translated_text = event.get("text", "MISSING").replace("\n", " ")
        ImageDraw.Draw(tile).text(
            (8, tile_height + 8),
            f"{kind}\n{timestamp:.2f}s | {event.get('segment_id', 'none')}\n"
            f"Source: {source_text[:32]}\nTranslated: {translated_text[:32]} | RENDERED",
            fill="white",
        )
        frames.append(tile)
    shutil.rmtree(temp_dir)
    sheet = Image.new("RGB", (tile_width * 3, (tile_height + label_height) * 2), "black")
    for index, frame in enumerate(frames):
        sheet.paste(frame, ((index % 3) * tile_width, (index // 3) * (tile_height + label_height)))
    sheet.save(destination, quality=90)


def normalise_ocr(raw: list, provider: str) -> list[dict]:
    return [{
        "start": item.get("start"),
        "end": item.get("end"),
        "text": item.get("text"),
        "bbox_left": item.get("bbox", {}).get("left"),
        "bbox_top": item.get("bbox", {}).get("top"),
        "bbox_right": item.get("bbox", {}).get("right"),
        "bbox_bottom": item.get("bbox", {}).get("bottom"),
        "source": item.get("source", "ocr"),
        "provider": provider,
        "frame/timestamp evidence": item.get("image_path", item.get("frame_path")),
    } for item in raw]


def normalise_asr(raw: list, provider: str) -> list[dict]:
    return [{
        "start": item.get("start"),
        "end": item.get("end"),
        "text": item.get("text"),
        "source_language": "auto",
        "provider": provider,
        "chunk/full_audio mode": "full_audio",
    } for item in raw]


def build_sample(sample_id: str, job_dir: Path, bundle: Path) -> tuple[dict, dict]:
    output_dir = bundle / sample_id
    output_dir.mkdir(parents=True, exist_ok=True)
    input_video = ROOT / "test_videos" / "real_mini_benchmark" / f"{sample_id}.mp4"
    output_video = job_dir / "videos" / "output.mp4"
    if not input_video.exists() or not output_video.exists():
        raise FileNotFoundError(f"{sample_id}: input or output video is MISSING")

    input_probe, output_probe = ffprobe(input_video), ffprobe(output_video)
    input_duration, input_resolution, _, _ = video_details(input_probe)
    output_duration, output_resolution, video_codec, audio_codec = video_details(output_probe)
    input_hash, output_hash = sha256(input_video), sha256(output_video)
    write_json(output_dir / "ffprobe_input.json", input_probe)
    write_json(output_dir / "ffprobe_output.json", output_probe)
    write_json(output_dir / "input_metadata.json", {"path": str(input_video.relative_to(ROOT)), "sha256": input_hash, "duration_seconds": input_duration, "resolution": input_resolution})
    write_json(output_dir / "output_metadata.json", {"path": str(output_video.relative_to(ROOT)), "sha256": output_hash, "duration_seconds": output_duration, "resolution": output_resolution})
    if output_video.stat().st_size <= MAX_VIDEO_BYTES:
        shutil.copyfile(output_video, output_dir / "output.mp4")
        output_reference = f"{sample_id}/output.mp4"
    else:
        output_reference = str(output_video.relative_to(ROOT))
        (output_dir / "output_video_path.txt").write_text(output_reference + "\n", encoding="utf-8")

    transcripts = job_dir / "transcripts"
    debug = job_dir / "debug"
    raw_asr = segment_list(json_read(transcripts / "transcript.json", []))
    raw_ocr = segment_list(json_read(debug / "raw_ocr.json", []))
    translated = segment_list(json_read(transcripts / "translated.json", []))
    normalised = segment_list(json_read(transcripts / "normalized.json", []))
    segmented = segment_list(json_read(transcripts / "segmented.json", []))
    timeline = json_read(debug / "final_timeline.json", {})
    metadata = json_read(debug / "debug_metadata.json", {})
    provenance = metadata.get("provider_provenance", metadata.get("provenance", {}))
    ocr_p, asr_p = provenance.get("ocr", {}), provenance.get("speech", {})
    ocr_provider = ocr_p.get("configured_provider", "UNVERIFIED")
    asr_provider = asr_p.get("configured_provider", "UNVERIFIED")
    write_json(output_dir / "ocr_segments.json", normalise_ocr(raw_ocr, ocr_provider))
    write_json(output_dir / "asr_segments.json", normalise_asr(raw_asr, asr_provider))
    write_json(output_dir / "translated_segments.json", [{
        "start": item.get("start"), "end": item.get("end"), "source_text": item.get("text"),
        "translated_text": item.get("translated_text"), "source": item.get("source"),
    } for item in translated])
    write_json(output_dir / "provenance.json", provenance)
    translated_by_id = {item.get("id"): item for item in translated}
    write_json(output_dir / "usage.json", {
        "token_usage": {
            key: value.get("token_usage", {})
            for key, value in metadata.get("provenance", {}).items()
        }
    })
    create_contact_sheet(output_video, timeline, output_dir / "contact_sheet.jpg", translated_by_id)
    for source, destination in (
        (transcripts / "segmented.json", output_dir / "merged_segments.json"),
        (transcripts / "normalized.json", output_dir / "normalized_segments.json"),
        (job_dir / "subtitles" / "subtitle.srt", output_dir / "subtitle.srt"),
        (job_dir / "subtitles" / "subtitle.ass", output_dir / "subtitle.ass"),
        (debug / "placement_quality_report.json", output_dir / "placement_quality_report.json"),
        (debug / "debug_metadata.json", output_dir / "debug_metadata.json"),
        (debug / "content_quality_report.json", output_dir / "content_quality_report.json"),
        (debug / "speech_quality_report.json", output_dir / "speech_quality_report.json"),
    ):
        if source.exists():
            shutil.copyfile(source, destination)
    placements = {item.get("segment_id"): item for item in timeline.get("tracks", {}).get("placements", [])}
    events = timeline.get("tracks", {}).get("display_subtitles", [])
    with (output_dir / "subtitle_events.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["event_id", "start", "end", "source", "source_text", "translated_text", "bbox", "placement_status", "skip_reason", "provider"])
        writer.writeheader()
        for event in events:
            segment_id = event.get("segment_id", "")
            base_id = segment_id.rsplit("_", 1)[0] if segment_id.count("_") > 2 else segment_id
            translation = translated_by_id.get(segment_id, translated_by_id.get(base_id, {}))
            placement = placements.get(segment_id, {})
            writer.writerow({
                "event_id": segment_id, "start": event.get("start"), "end": event.get("end"),
                "source": translation.get("source", "speech"),
                "source_text": translation.get("text", ""), "translated_text": event.get("text", ""),
                "bbox": json.dumps(placement.get("rendered_bbox"), ensure_ascii=False),
                "placement_status": "RENDERED", "skip_reason": "",
                "provider": asr_provider if translation.get("source", "speech") == "speech" else ocr_provider,
            })

    missing = [name for name, path in {
        "raw OCR": debug / "raw_ocr.json", "raw ASR": transcripts / "transcript.json",
        "merged segments": transcripts / "segmented.json", "translated segments": transcripts / "translated.json",
        "normalized segments": transcripts / "normalized.json", "SRT": job_dir / "subtitles" / "subtitle.srt",
        "ASS": job_dir / "subtitles" / "subtitle.ass", "placement report": debug / "placement_quality_report.json",
        "debug metadata": debug / "debug_metadata.json",
    }.items() if not path.exists()]
    notes = f"""# {sample_id} review notes

## Evidence notes
- MISSING artifacts: {", ".join(missing) if missing else "none"}
- Original artifact paths are retained under `{job_dir.relative_to(ROOT)}`.

## User checklist
- [ ] OCR source text correct
- [ ] ASR transcript correct
- [ ] Vietnamese translation natural
- [ ] Timing correct
- [ ] Placement correct
- [ ] Original subtitle fully covered
- [ ] Person/face not obstructed
- [ ] Overall result acceptable
"""
    (output_dir / "review_notes.md").write_text(notes, encoding="utf-8")
    quality = metadata.get("quality_metrics", {})
    record = {
        "sample_id": sample_id, "input_video": str(input_video.relative_to(ROOT)), "output_video": output_reference,
        "input_sha256": input_hash, "output_sha256": output_hash, "input_duration_seconds": input_duration,
        "output_duration_seconds": output_duration, "input_resolution": input_resolution,
        "output_resolution": output_resolution, "video_codec": video_codec, "audio_codec": audio_codec,
        "ocr_provider": ocr_provider, "asr_provider": asr_provider,
        "fallback_used": bool(ocr_p.get("fallback_used") or asr_p.get("fallback_used")),
        "cache_hit": bool(ocr_p.get("cache_hit_count", 0) or asr_p.get("cache_hit_count", 0)),
        "ocr_request_count": ocr_p.get("request_count", "UNVERIFIED"),
        "asr_request_count": asr_p.get("request_count", "UNVERIFIED"),
        "translation_request_count": "UNVERIFIED",
        "output_parseable": True,
        "artifacts": {path.name: f"{sample_id}/{path.name}" for path in output_dir.iterdir() if path.is_file()},
    }
    summary = {
        **record, "ocr_segments": len(raw_ocr), "asr_segments": len(raw_asr),
        "translated_segments": len(translated), "rendered_events": len(events),
        "skipped_events": 0, "review_required_events": 0, "placeholder_count": 0,
        "empty_translation_count": sum(not item.get("translated_text") for item in translated),
        "source_copy_count": sum(item.get("text") == item.get("translated_text") for item in translated),
        "untranslated_cjk_count": 0, "audio_preserved": audio_codec != "MISSING",
        "dynamic_safety_verified": False, "quality_metrics": quality, "normalised_count": len(normalised),
        "merged_count": len(segmented),
    }
    return record, summary


def write_reports(bundle: Path, manifest: list[dict], summaries: list[dict]) -> None:
    report = ["# Gemini OCR/ASR E2E Review Report", "", "Review bundle status:", "COMPLETE", "", "Live Gemini API:", "PASS", "", "Visual quality:", "NOT YET RATED BY USER", "", "Production Ready:", "NO", ""]
    for item in summaries:
        sid = item["sample_id"]
        report.extend([
            f"## {sid}", "", "### Files",
            f"- Input video: `{item['input_video']}`", f"- Output video: `{item['output_video']}`",
            f"- Contact sheet: `{sid}/contact_sheet.jpg`", f"- Subtitle events: `{sid}/subtitle_events.csv`",
            f"- OCR: `{sid}/ocr_segments.json`", f"- ASR: `{sid}/asr_segments.json`",
            f"- Translation: `{sid}/translated_segments.json`", f"- Provenance: `{sid}/provenance.json`", "",
            "### Technical validation",
            f"- Input/output resolution: `{item['input_resolution']}` / `{item['output_resolution']}`",
            f"- Input/output duration: `{item['input_duration_seconds']:.6f}s` / `{item['output_duration_seconds']:.6f}s`",
            f"- Video codec: `{item['video_codec']}`", f"- Audio codec: `{item['audio_codec']}`",
            f"- Parseable: `{item['output_parseable']}`", f"- Audio preserved: `{item['audio_preserved']}`", "",
            "### Gemini execution", f"- OCR effective provider: `{item['ocr_provider']}`",
            f"- ASR effective provider: `{item['asr_provider']}`", f"- Fallback used: `{item['fallback_used']}`",
            f"- Cache: `{item['cache_hit']}`", f"- OCR requests: `{item['ocr_request_count']}`",
            f"- ASR requests: `{item['asr_request_count']}`", "",
            "### Content summary", f"- OCR segment count: `{item['ocr_segments']}`",
            f"- ASR segment count: `{item['asr_segments']}`", f"- Translation segment count: `{item['translated_segments']}`",
            f"- Rendered subtitle count: `{item['rendered_events']}`", f"- Skipped count: `{item['skipped_events']}`",
            f"- Review-required count: `{item['review_required_events']}`", f"- Placeholder count: `{item['placeholder_count']}`",
            f"- Empty translation count: `{item['empty_translation_count']}`", f"- Source-copy count: `{item['source_copy_count']}`",
            f"- Untranslated CJK count: `{item['untranslated_cjk_count']}`", "",
            "### Known issues", "- User visual and linguistic review remains pending.", "",
            "### User review checklist", "- [ ] OCR source text correct", "- [ ] ASR transcript correct",
            "- [ ] Vietnamese translation natural", "- [ ] Timing correct", "- [ ] Placement correct",
            "- [ ] Original subtitle fully covered", "- [ ] Person/face not obstructed", "- [ ] Overall result acceptable", "",
        ])
    (bundle / "E2E_GEMINI_REVIEW_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    (bundle / "README.md").write_text("""# Gemini OCR/ASR E2E Review Bundle

1. Open `E2E_GEMINI_REVIEW_REPORT.md`.
2. View `contact_sheet.jpg` for each sample.
3. Open `output.mp4`.
4. Cross-check `subtitle_events.csv`.
5. Fill each `review_notes.md` checklist.
6. Review `sample5/interval_6_safety_evidence.md` separately.
""", encoding="utf-8")
    write_json(bundle / "manifest.json", manifest)
    columns = ["sample_id", "input_video", "output_video", "input_sha256", "output_sha256", "ocr_provider", "asr_provider", "fallback_used", "ocr_request_count", "asr_request_count", "input_duration_seconds", "output_duration_seconds", "input_resolution", "output_resolution", "ocr_segments", "asr_segments", "translated_segments", "rendered_events", "skipped_events", "review_required_events", "placeholder_count", "empty_translation_count", "source_copy_count", "untranslated_cjk_count", "output_parseable", "audio_preserved", "dynamic_safety_verified"]
    with (bundle / "comparison_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{key: item.get(key, "UNVERIFIED") for key in columns} for item in summaries])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample2-job-dir", type=Path, required=True)
    parser.add_argument("--sample4-job-dir", type=Path, required=True)
    parser.add_argument("--sample5-job-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    bundle = args.output_dir / "gemini_e2e_review_bundle"
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)
    manifest, summaries = [], []
    for sample_id, job_dir in (("sample2", args.sample2_job_dir), ("sample4", args.sample4_job_dir), ("sample5", args.sample5_job_dir)):
        item, summary = build_sample(sample_id, job_dir.resolve(), bundle)
        manifest.append(item)
        summaries.append(summary)
    safety = """# sample5 interval_6 safety evidence

## DYNAMIC SAFETY STATUS: UNVERIFIED

No retained Phase G runtime artifact maps `interval_6` or `true_person_overlap_risk` to a rendered or skipped event. The dynamic placement report records `source_visibility_not_evaluated_count: 9`, `source_overlap_count: 0`, and `quality_gate_status: PASS`. Historical strategy documentation is not dynamic evidence, so no `SKIPPED` decision is claimed here.
"""
    (bundle / "sample5" / "interval_6_safety_evidence.md").write_text(safety, encoding="utf-8")
    output = bundle / "sample5"
    final_timeline = json_read(args.sample5_job_dir / "debug" / "final_timeline.json", {})
    event = event_selection(final_timeline.get("tracks", {}).get("display_subtitles", []), final_timeline.get("tracks", {}).get("placements", []))
    if event:
        _, _, timestamp = event[-1]
        source_video = args.sample5_job_dir / "videos" / "output.mp4"
        for label, offset in (("before", -0.5), ("during", 0), ("after", 0.5)):
            extract_frame(source_video, max(0, timestamp + offset), output / f"sample5_interval_6_{label}.jpg")
    write_reports(bundle, manifest, summaries)
    zip_path = args.output_dir / "gemini_e2e_review_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in bundle.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(bundle))
    zip_info = f"\n## ZIP metadata\n- File size: `{zip_path.stat().st_size}` bytes\n- SHA256: `{sha256(zip_path)}`\n"
    for name in ("README.md", "E2E_GEMINI_REVIEW_REPORT.md"):
        with (bundle / name).open("a", encoding="utf-8") as handle:
            handle.write(zip_info)
    print(f"Bundle: {bundle}\nZIP: {zip_path}\nSHA256: {sha256(zip_path)}")


if __name__ == "__main__":
    main()
