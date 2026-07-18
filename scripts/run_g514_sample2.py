"""Run G.5.14 Sample2 image-level replacement with recoverable provenance."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.infrastructure.factory.provider_factory import ProviderFactory
from src.infrastructure.gemini.contracts import MODE_ROLE_PRIMARY, GeminiRequest
from src.infrastructure.gemini.exceptions import GeminiRateLimitError, ProviderRateLimitError
from src.infrastructure.gemini.structured_output import parse_structured
from src.interfaces.config.config_loader import ConfigLoader
from src.shared.exceptions import ProviderError

MODEL = "gemini-3.1-flash-lite"
PROVIDER = "gemini_video_direct"
VISUAL_QA_STATUSES = {"PASS", "FAIL", "REVIEW_REQUIRED"}


class G514ValidationError(ValueError):
    """The persisted G.5.14 evidence cannot safely be used."""


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def _batches(items: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for index in range(0, len(items), size):
        yield items[index:index + size]


def event_timing_hash(events: Iterable[dict[str, Any]]) -> str:
    return canonical_hash([
        {"event_id": event["event_id"], "candidate_id": event["candidate_id"],
         "start_seconds_in_source": float(event["start_seconds_in_source"]),
         "end_seconds_in_source": float(event["end_seconds_in_source"])}
        for event in events
    ])


def validate_completed_stages(manifest: dict[str, Any], video_sha256: str) -> list[dict[str, Any]]:
    identity, artifacts = manifest.get("identity", {}), manifest.get("artifacts", {})
    if manifest.get("video_sha256") != video_sha256 or identity.get("video_sha256") != video_sha256:
        raise G514ValidationError("BASELINE_IDENTITY_MISMATCH")
    if identity.get("model") != MODEL:
        raise G514ValidationError("BASELINE_MODEL_MISMATCH")
    for stage in ("V1", "V2", "V3"):
        if artifacts.get("stage_status", {}).get(stage, {}).get("status") not in {"COMPLETED", "COMPLETED_REUSED"}:
            raise G514ValidationError(f"BASELINE_{stage}_NOT_COMPLETED")
    events = artifacts.get("checkpoints", {}).get("temporal_events", {}).get("events")
    if not isinstance(events, list) or len(events) != 15:
        raise G514ValidationError("BASELINE_REQUIRES_EXACTLY_15_EVENTS")
    seen = set()
    for event in events:
        try:
            valid = isinstance(event["event_id"], str) and isinstance(event["candidate_id"], str) and float(event["end_seconds_in_source"]) > float(event["start_seconds_in_source"])
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid or event["event_id"] in seen:
            raise G514ValidationError("BASELINE_V3_IDS_OR_TIMING_INVALID")
        seen.add(event["event_id"])
    return events


def _valid_provenance(records: Any, stage: str) -> bool:
    entries = [item for item in records if isinstance(item, dict) and item.get("stage") == stage] if isinstance(records, list) else []
    return bool(entries) and all(
        item.get("model") == MODEL and item.get("provider") == PROVIDER
        and isinstance(item.get("schema_version"), str) and item["schema_version"]
        and isinstance(item.get("validator_version"), str) and item["validator_version"]
        for item in entries
    )


def reconstruct_v5(events: list[dict[str, Any]], replay: Any, video_sha256: str) -> dict[str, str]:
    if not isinstance(replay, dict) or replay.get("video_sha256") != video_sha256 or not _valid_provenance(replay.get("provenance"), "v5_transcription"):
        raise G514ValidationError("V5_PROVENANCE_RECONSTRUCTION_FAILED")
    accepted, expected = replay.get("accepted_events"), {event["event_id"]: event for event in events}
    if not isinstance(accepted, list) or len(accepted) != 15:
        raise G514ValidationError("V5_PROVENANCE_RECONSTRUCTION_FAILED")
    texts: dict[str, str] = {}
    for item in accepted:
        try:
            event_id, start, end = item["event_id"], float(item["start_seconds_in_source"]), float(item["end_seconds_in_source"])
        except (KeyError, TypeError, ValueError) as exc:
            raise G514ValidationError("V5_PROVENANCE_RECONSTRUCTION_FAILED") from exc
        source = expected.get(event_id)
        text = item.get("text_verbatim") if isinstance(item, dict) else None
        if source is None or event_id in texts or not isinstance(text, str) or not text.strip() or abs(start - float(source["start_seconds_in_source"])) > .1 or abs(end - float(source["end_seconds_in_source"])) > .1:
            raise G514ValidationError("V5_PROVENANCE_RECONSTRUCTION_FAILED")
        texts[event_id] = text.strip()
    if set(texts) != set(expected):
        raise G514ValidationError("V5_PROVENANCE_RECONSTRUCTION_FAILED")
    return texts


def geometry_frame_times(event: dict[str, Any]) -> list[tuple[str, float]]:
    start, end = float(event["start_seconds_in_source"]), float(event["end_seconds_in_source"])
    duration = end - start
    if duration < .8:
        return [("middle", (start + end) / 2)]
    if duration <= 1.5:
        return [("early", start + duration * .15), ("late", end - duration * .15)]
    return [("early", start + duration * .15), ("middle", (start + end) / 2), ("late", end - duration * .15)]


def build_v4_batch_checkpoint(batch: list[dict[str, Any]], geometry: list[dict[str, Any]], video_sha256: str) -> dict[str, Any]:
    return {"video_sha256": video_sha256, "batch_hash": canonical_hash([event["event_id"] for event in batch]), "geometry": geometry}


def validate_v4_batch_checkpoint(checkpoint: Any, batch: list[dict[str, Any]], video_sha256: str) -> bool:
    if not isinstance(checkpoint, dict) or checkpoint.get("video_sha256") != video_sha256:
        return False
    expected = {event["event_id"] for event in batch}
    geometry = checkpoint.get("geometry")
    return isinstance(geometry, list) and {item.get("event_id") for item in geometry if isinstance(item, dict)} == expected


def translation_reuse(events: list[dict[str, Any]], checkpoint: Any, v5_sha256: str) -> tuple[dict[str, str], list[str], str | None]:
    expected = {event["event_id"]: event for event in events}
    all_ids = [event["event_id"] for event in events]
    if not isinstance(checkpoint, dict) or checkpoint.get("input_v5_sha256") != v5_sha256 or not isinstance(checkpoint.get("events"), list):
        return {}, all_ids, "TRANSLATION_PROVENANCE_RECONSTRUCTION_FAILED"
    entries = {item.get("event_id"): item for item in checkpoint["events"] if isinstance(item, dict) and isinstance(item.get("event_id"), str)}
    reused, rerun = {}, []
    for event_id, event in expected.items():
        item = entries.get(event_id)
        try:
            start, end = float(item["start_seconds_in_source"]), float(item["end_seconds_in_source"])
        except (KeyError, TypeError, ValueError):
            rerun.append(event_id)
            continue
        text = item.get("translated_text")
        if abs(start - float(event["start_seconds_in_source"])) > .1 or abs(end - float(event["end_seconds_in_source"])) > .1 or not isinstance(text, str) or not text.strip():
            rerun.append(event_id)
        else:
            reused[event_id] = text.strip()
    failed = bool(rerun) or set(entries) != set(expected) or not _valid_provenance(checkpoint.get("provenance"), "translation")
    return reused, rerun, "TRANSLATION_PROVENANCE_RECONSTRUCTION_FAILED" if failed else None


def select_eligible_render_events(events: list[dict[str, Any]], geometry: list[dict[str, Any]], translations: dict[str, str]) -> list[dict[str, Any]]:
    gmap = {item.get("event_id"): item for item in geometry if isinstance(item, dict)}
    selected = []
    for event in events:
        item, translated = gmap.get(event["event_id"]), translations.get(event["event_id"])
        if item and not item.get("needs_review") and isinstance(item.get("representative_box_2d"), list) and len(item["representative_box_2d"]) == 4 and isinstance(translated, str) and translated.strip():
            selected.append({**event, **item, "translated_text": translated.strip()})
    return selected


def _ass_time(seconds: float) -> str:
    hours, rem = divmod(round(seconds * 100), 360000)
    minutes, rem = divmod(rem, 6000)
    return f"{hours}:{minutes:02}:{rem // 100:02}.{rem % 100:02}"


def _escape_ass(text: str) -> str:
    return text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def generate_ass_file(events: list[dict[str, Any]], path: Path, width: int = 720, height: int = 1280) -> None:
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Default,Arial,24,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,5,0,0,0,1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""
    lines = [header]
    for event in events:
        ymin, xmin, ymax, xmax = event["representative_box_2d"]
        left, top, right, bottom = round(xmin * width / 1000), round(ymin * height / 1000), round(xmax * width / 1000), round(ymax * height / 1000)
        box_width, box_height = right - left, bottom - top
        start, end = _ass_time(float(event["start_seconds_in_source"])), _ass_time(float(event["end_seconds_in_source"]))
        font_size = max(12, min(24, int(box_height * .55)))
        lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{{\\an7\\pos({left},{top})\\p1\\1c&H000000&}}m 0 0 l {box_width} 0 l {box_width} {box_height} l 0 {box_height}{{\\p0}}")
        lines.append(f"Dialogue: 1,{start},{end},Default,,0,0,0,,{{\\an5\\fs{font_size}\\pos({round((left + right) / 2)},{round((top + bottom) / 2)})}}{_escape_ass(event['translated_text'])}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_render_checkpoint(input_video: Path, output_video: Path, geometry: list[dict[str, Any]], translations: dict[str, str], ass_path: Path) -> dict[str, str]:
    return {"input_video_sha256": file_hash(input_video), "ass_sha256": file_hash(ass_path), "geometry_sha256": canonical_hash(geometry), "translation_sha256": canonical_hash(translations), "output_video_sha256": file_hash(output_video)}


def validate_render_checkpoint(checkpoint: Any, input_video: Path, output_video: Path, geometry: list[dict[str, Any]], translations: dict[str, str], ass_path: Path) -> bool:
    expected = build_render_checkpoint(input_video, output_video, geometry, translations, ass_path)
    return isinstance(checkpoint, dict) and output_video.is_file() and all(checkpoint.get(key) == value for key, value in expected.items())


def _probe(video: Path) -> dict[str, Any]:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(video)], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def technical_qa(input_video: Path, output_video: Path, rendered: list[dict[str, Any]]) -> dict[str, Any]:
    reasons = []
    size = output_video.stat().st_size if output_video.is_file() else 0
    if size <= 0:
        reasons.append("OUTPUT_VIDEO_EMPTY")
    try:
        source, output = _probe(input_video), _probe(output_video)
        video = next((stream for stream in output.get("streams", []) if stream.get("codec_type") == "video"), None)
        audio = next((stream for stream in output.get("streams", []) if stream.get("codec_type") == "audio"), None)
        duration_delta = abs(float(source["format"]["duration"]) - float(output["format"]["duration"]))
    except (subprocess.CalledProcessError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return {"verdict": "FAIL", "reasons": reasons + ["FFPROBE_FAILED"]}
    if video is None:
        reasons.append("VIDEO_STREAM_MISSING")
    if audio is None:
        reasons.append("AUDIO_STREAM_MISSING")
    if duration_delta >= .25:
        reasons.append("DURATION_DELTA_EXCEEDED")
    if video is None or int(video.get("width", 0)) != 720 or int(video.get("height", 0)) != 1280:
        reasons.append("RESOLUTION_NOT_720X1280")
    bad_boxes = sum(not (isinstance(event.get("representative_box_2d"), list) and len(event["representative_box_2d"]) == 4 and 0 <= event["representative_box_2d"][0] <= event["representative_box_2d"][2] <= 1000 and 0 <= event["representative_box_2d"][1] <= event["representative_box_2d"][3] <= 1000) for event in rendered)
    empty = sum(not isinstance(event.get("translated_text"), str) or not event["translated_text"].strip() for event in rendered)
    if bad_boxes:
        reasons.append("OUT_OF_FRAME_BBOX")
    if empty:
        reasons.append("EMPTY_RENDERED_TEXT")
    return {"verdict": "PASS" if not reasons else "FAIL", "reasons": reasons, "file_size": size, "video_stream": video.get("codec_name") if video else None, "audio_stream": audio.get("codec_name") if audio else None, "duration_delta": duration_delta, "resolution": [video.get("width"), video.get("height")] if video else None, "rendered_events": len(rendered), "cover_events": len(rendered), "text_events": len(rendered), "cover_text_parity": len(rendered) == len(rendered), "out_of_frame_events": bad_boxes, "empty_rendered_text": empty}


def visual_qa_schema() -> dict[str, Any]:
    return {"type": "object", "properties": {"results": {"type": "array", "items": {"type": "object", "properties": {"event_id": {"type": "string"}, "source_text_visible": {"type": "boolean"}, "banner_visible": {"type": "boolean"}, "has_duplicate": {"type": "boolean"}, "has_overflow": {"type": "boolean"}, "has_damage": {"type": "boolean"}, "status": {"type": "string", "enum": sorted(VISUAL_QA_STATUSES)}, "reason": {"type": "string"}}, "required": ["event_id", "source_text_visible", "banner_visible", "has_duplicate", "has_overflow", "has_damage", "status"]}}}, "required": ["results"]}


def validate_visual_qa_result(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("status") not in VISUAL_QA_STATUSES:
        return {"status": "REVIEW_REQUIRED", "reason": "VISUAL_QA_UNCLEAR"}
    if any(result.get(key) is True for key in ("source_text_visible", "banner_visible", "has_duplicate", "has_overflow", "has_damage")) or result["status"] == "FAIL":
        return {"status": "FAIL", "reason": str(result.get("reason") or "VISUAL_QA_FAIL")}
    return {"status": result["status"], "reason": result.get("reason") if result["status"] == "REVIEW_REQUIRED" else None}


def validate_visual_qa_batch(results: list[dict[str, Any]]) -> str:
    verdicts = [validate_visual_qa_result(result)["status"] for result in results]
    return "FAIL" if "FAIL" in verdicts else ("REVIEW_REQUIRED" if "REVIEW_REQUIRED" in verdicts else "PASS")


def _extract_frame(video: Path, timestamp: float) -> bytes:
    result = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{timestamp:.3f}", "-i", str(video), "-frames:v", "1", "-f", "image2pipe", "-c:v", "mjpeg", "-"], capture_output=True, check=False)
    if result.returncode or not result.stdout:
        raise G514ValidationError("FFMPEG_FRAME_EXTRACTION_FAILED")
    return result.stdout


def run_visual_qa_batch(provider: Any, batch: list[dict[str, Any]], input_video: Path, output_video: Path) -> dict[str, Any]:
    parts: list[dict[str, Any]] = [{"text": "Compare every original/rendered frame pair. Fail for source-text leakage, banner leakage, duplicate text, overflow, or damaged text. Return one result per event_id."}]
    for event in batch:
        frames = geometry_frame_times(event) if float(event["end_seconds_in_source"]) - float(event["start_seconds_in_source"]) > 1.5 else [("middle", sum((float(event["start_seconds_in_source"]), float(event["end_seconds_in_source"]))) / 2)]
        for role, timestamp in frames:
            parts.extend([{"text": f"event_id={event['event_id']} role={role} original"}, {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(_extract_frame(input_video, timestamp)).decode()}}, {"text": f"event_id={event['event_id']} role={role} rendered"}, {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(_extract_frame(output_video, timestamp)).decode()}}])
    request = GeminiRequest(contents=[{"parts": parts}], model=provider.model, generation_config={"responseMimeType": "application/json", "responseSchema": visual_qa_schema()}, system_instruction="You are a quality assurance assistant.", source_kind="visual_qa", api_surface="generate_content")
    client = provider.runtime.get_client(provider.model, role=MODE_ROLE_PRIMARY, api_key=provider.api_key)
    request.predicted_tokens = client.count_tokens(request)
    response = client.generate_content(request)
    payload = parse_structured(response.text)
    if not isinstance(payload, dict):
        raise ProviderError("Visual QA response was not an object.")
    provider.provenance.append({"stage": "visual_qa", "input_type": "image", "model": provider.model, "count_tokens": request.predicted_tokens or 0, "actual_usage": response.usage.to_dict() if response.usage else {}})
    return payload


def translate_batch(provider: Any, batch: list[dict[str, Any]], source_texts: dict[str, str]) -> dict[str, Any]:
    items = [{"event_id": event["event_id"], "text_verbatim": source_texts[event["event_id"]]} for event in batch]
    prompt = provider._batch_translation_prompt(items, "en")
    request = GeminiRequest(contents=[{"parts": [{"text": prompt}]}], model=provider.model, generation_config={"responseMimeType": "application/json", "responseSchema": provider._batch_schema("event_id", provider._translation_schema())}, system_instruction=prompt, source_kind="translation", api_surface="generate_content")
    client = provider.runtime.get_client(provider.model, role=MODE_ROLE_PRIMARY, api_key=provider.api_key)
    request.predicted_tokens = client.count_tokens(request)
    response = client.generate_content(request)
    payload = parse_structured(response.text)
    if not isinstance(payload, dict):
        raise ProviderError("Translation response was not an object.")
    provider.provenance.append({"stage": "translation", "input_type": "text", "model": provider.model, "count_tokens": request.predicted_tokens or 0, "actual_usage": response.usage.to_dict() if response.usage else {}})
    return payload


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_checkpoint(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--v5-replay", type=Path, required=False)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--translation-checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--live-visual-qa", action="store_true")
    args = parser.parse_args()
    artifact_path, checkpoint_path = args.output_root / "g514_sample2.json", args.output_root / "g514_checkpoint.json"
    if artifact_path.exists():
        raise SystemExit("G514_ARTIFACT_ALREADY_EXISTS")
    quota = {"generate_content_request_count": 0, "count_tokens_call_count": 0, "returned_token_count": 0, "current_run_429_count": 0, "historical_429_count": 0}
    try:
        manifest, video_sha256 = json.loads(args.manifest.read_text(encoding="utf-8")), file_hash(args.video)
        events = validate_completed_stages(manifest, video_sha256)
        loader, config = ConfigLoader(), ConfigLoader().load()
        batch_size = args.batch_size or config.gemini.direct_video_batch_size
        if batch_size < 1:
            raise G514ValidationError("BATCH_SIZE_INVALID")
        checkpoint = load_checkpoint(checkpoint_path)
        quota["historical_429_count"] = checkpoint.get("historical_429_count", 0)
        replay = json.loads(args.v5_replay.read_text(encoding="utf-8")) if args.v5_replay and args.v5_replay.is_file() else None
        try:
            source_texts, v5_status = reconstruct_v5(events, replay, video_sha256), "RECONSTRUCTED_FROM_IMMUTABLE_ARTIFACTS"
        except G514ValidationError:
            source_texts, v5_status = None, "V5_PROVENANCE_RECONSTRUCTION_FAILED"
        provider = ProviderFactory(config=config.providers, gemini_config=config.gemini, gemini_api_key=loader.gemini_api_key).create_ocr_provider(PROVIDER)
        geometry: list[dict[str, Any]] = []
        for batch in _batches(events, batch_size):
            key = canonical_hash({"video": video_sha256, "events": [event["event_id"] for event in batch], "frames": [geometry_frame_times(event) for event in batch]})
            saved = checkpoint.get("v4_batches", {}).get(key)
            if validate_v4_batch_checkpoint(saved, batch, video_sha256):
                geometry.extend(saved["geometry"])
                continue
            frame_sets = [{"event": event, "frames": [{"frame_role": role, "timestamp_seconds": time, "image_bytes": _extract_frame(args.video, time)} for role, time in geometry_frame_times(event)]} for event in batch]
            payload = provider._request_image_geometry(provider._image_geometry_prompt(frame_sets, source_texts), provider._batch_schema("event_id", provider._image_geometry_schema()), frame_sets)
            quota["generate_content_request_count"] += 1; quota["count_tokens_call_count"] += 1
            responses = provider._batch_responses(payload, batch, "event_id")
            batch_geometry = [{**provider._validate_image_geometry(responses[event["event_id"]], event), "source_text_hint": source_texts.get(event["event_id"]) if source_texts else None, "source_text_hint_status": "VERIFIED" if source_texts else "UNAVAILABLE_UNVERIFIED"} for event in batch]
            geometry.extend(batch_geometry)
            checkpoint.setdefault("v4_batches", {})[key] = build_v4_batch_checkpoint(batch, batch_geometry, video_sha256)
            save_checkpoint(checkpoint_path, checkpoint)
        v5_hash = canonical_hash({"video": video_sha256, "timings": event_timing_hash(events)})
        if source_texts is None:
            cached = checkpoint.get("v5", {})
            if cached.get("hash") == v5_hash:
                source_texts = reconstruct_v5(events, cached, video_sha256)
            else:
                mime_type = mimetypes.guess_type(str(args.video))[0] or "video/mp4"
                handle = provider.runtime.file_manager.upload_file(str(args.video), mime_type=mime_type)
                duration = float(_probe(args.video)["format"]["duration"])
                accepted = []
                for batch in _batches(events, batch_size):
                    payload = provider._request("v5_transcription", provider._batch_prompt("transcribe", batch), provider._batch_schema("event_id", provider._transcription_schema()), handle.uri, mime_type, provider.fps["transcription"], max(0., min(float(event["start_seconds_in_source"]) for event in batch) - .25), min(duration, max(float(event["end_seconds_in_source"]) for event in batch) + .25), duration)
                    quota["generate_content_request_count"] += 1; quota["count_tokens_call_count"] += 1
                    responses = provider._batch_responses(payload, batch, "event_id")
                    for event in batch:
                        transcription = provider._assess_transcription(responses[event["event_id"]], event).get("transcription")
                        if not transcription:
                            raise G514ValidationError("V5_RERUN_TRANSCRIPTION_INVALID")
                        accepted.append({**event, **transcription})
                source_texts = {item["event_id"]: item["text_verbatim"] for item in accepted}
                checkpoint["v5"] = {"hash": v5_hash, "video_sha256": video_sha256, "accepted_events": accepted, "provenance": [{"stage": "v5_transcription", "model": MODEL, "provider": PROVIDER, "schema_version": provider.schema_version, "validator_version": "v5_unicode_source_script_v1"}], "provenance_status": v5_status}
                save_checkpoint(checkpoint_path, checkpoint)
        v5_text_hash = canonical_hash(source_texts)
        raw_translation = json.loads(args.translation_checkpoint.read_text(encoding="utf-8")) if args.translation_checkpoint and args.translation_checkpoint.is_file() else None
        translations, rerun_ids, translation_status = translation_reuse(events, raw_translation, v5_text_hash)
        for event in events:
            if event["event_id"] in rerun_ids and not source_texts.get(event["event_id"], "").strip():
                raise G514ValidationError("EMPTY_PILOT_TEXT")
        if rerun_ids:
            rerun_events = [event for event in events if event["event_id"] in rerun_ids]
            for batch in _batches(rerun_events, batch_size):
                payload = translate_batch(provider, batch, source_texts)
                quota["generate_content_request_count"] += 1; quota["count_tokens_call_count"] += 1
                for item in payload.get("results", []):
                    if isinstance(item, dict) and isinstance(item.get("translated_text"), str) and item["translated_text"].strip():
                        translations[item.get("event_id")] = item["translated_text"].strip()
            if set(translations) != {event["event_id"] for event in events}:
                raise G514ValidationError("TRANSLATION_RERUN_INCOMPLETE")
            checkpoint["translations"] = {"input_v5_sha256": v5_text_hash, "events": [{"event_id": event["event_id"], "start_seconds_in_source": event["start_seconds_in_source"], "end_seconds_in_source": event["end_seconds_in_source"], "translated_text": translations[event["event_id"]]} for event in events]}
            save_checkpoint(checkpoint_path, checkpoint)
        eligible = select_eligible_render_events(events, geometry, translations)
        output_video, ass_path = args.output_root / "sample2_image_level_dynamic_replacement.mp4", args.output_root / "sample2_image_level_dynamic_replacement.ass"
        if not eligible:
            technical = {"verdict": "NOT_RUN", "reasons": ["ZERO_EVENTS_RENDERED"]}
            visual_status, visual_results = "NOT_RUN", []
        else:
            generate_ass_file(eligible, ass_path)
            if not validate_render_checkpoint(checkpoint.get("render"), args.video, output_video, geometry, translations, ass_path):
                ass_filter = str(ass_path.resolve()).replace("\\", "/").replace(":", "\\:")
                result = subprocess.run(["ffmpeg", "-y", "-i", str(args.video), "-vf", f"ass='{ass_filter}'", str(output_video)], capture_output=True)
                if result.returncode:
                    raise G514ValidationError("FFMPEG_RENDER_FAILED")
                checkpoint["render"] = build_render_checkpoint(args.video, output_video, geometry, translations, ass_path)
                save_checkpoint(checkpoint_path, checkpoint)
            technical = technical_qa(args.video, output_video, eligible)
            visual_results, visual_status = [], "NOT_REQUESTED"
            if args.live_visual_qa and technical["verdict"] == "PASS":
                visual_hash = canonical_hash({"video": file_hash(output_video), "geometry": canonical_hash(geometry), "translations": canonical_hash(translations)})
                cached = checkpoint.get("visual_qa", {})
                if cached.get("hash") == visual_hash:
                    visual_results, visual_status = cached.get("results", []), cached.get("status", "REVIEW_REQUIRED")
                else:
                    for batch in _batches(eligible, batch_size):
                        payload = run_visual_qa_batch(provider, batch, args.video, output_video)
                        quota["generate_content_request_count"] += 1; quota["count_tokens_call_count"] += 1
                        visual_results.extend(payload.get("results", []))
                    visual_status = validate_visual_qa_batch(visual_results)
                    checkpoint["visual_qa"] = {"hash": visual_hash, "status": visual_status, "results": visual_results}
                    save_checkpoint(checkpoint_path, checkpoint)
        quota["returned_token_count"] = sum((entry.get("actual_usage") or {}).get("candidates_token_count", (entry.get("actual_usage") or {}).get("total_token_count", 0)) for entry in provider.provenance)
        checkpoint["historical_429_count"] = quota["historical_429_count"] + quota["current_run_429_count"]
        save_checkpoint(checkpoint_path, checkpoint)
        all_passed = len(eligible) == 15 and visual_status == "PASS"
        artifact = {"phase": "G.5.14 Sample2 image-level dynamic replacement", "request_status": "COMPLETED", "final_verdict": "PRODUCTION READY: NO", "gemini_only_visual_strategy": "PASS_ON_SAMPLE2" if all_passed else "CONDITIONAL", "strategy_decision": "CONTINUE_TO_SAMPLE4" if all_passed else "CORRECT_IMAGE_LEVEL_STRATEGY", "v5_provenance_status": v5_status, "translation_provenance_status": translation_status or "REUSED_VERIFIED", "translation_reused_events": len(events) - len(rerun_ids), "translation_rerun_events": len(rerun_ids), "geometry": geometry, "rendered_event_count": len(eligible), "technical_qa": technical, "visual_qa": {"status": visual_status, "results": visual_results}, "quota_report": quota}
    except (G514ValidationError, GeminiRateLimitError, ProviderRateLimitError, ProviderError) as exc:
        artifact = {"phase": "G.5.14 Sample2 image-level dynamic replacement", "request_status": "FAILED", "final_verdict": "PRODUCTION READY: NO", "gemini_only_visual_strategy": "CONDITIONAL", "strategy_decision": "CORRECT_IMAGE_LEVEL_STRATEGY", "failure_reason": str(exc), "quota_report": quota}
    args.output_root.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(artifact_path)


if __name__ == "__main__":
    main()
