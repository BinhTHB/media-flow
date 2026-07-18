"""Run or plan G.5.13 Sample2 image-level geometry validation."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.infrastructure.factory.provider_factory import ProviderFactory
from src.interfaces.config.config_loader import ConfigLoader
from src.infrastructure.gemini.contracts import MODE_ROLE_PRIMARY, GeminiRequest
from src.infrastructure.gemini.structured_output import parse_structured
from src.infrastructure.gemini.exceptions import GeminiRateLimitError, ProviderRateLimitError
from src.shared.exceptions import ProviderError

MODEL = "gemini-3.1-flash-lite"
PROVIDER = "gemini_video_direct"
RENDER_NOT_APPLICABLE = "NOT_APPLICABLE_LOCAL_RENDER"
VISUAL_QA_STATUSES = {"PASS", "FAIL", "REVIEW_REQUIRED"}


class G513ValidationError(ValueError):
    """A persisted G.5.13 input is not safe to reuse."""


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def file_hash(path: Path) -> str:
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def event_timing_hash(events: Iterable[dict[str, Any]]) -> str:
    return canonical_hash([
        {
            "event_id": event["event_id"],
            "candidate_id": event["candidate_id"],
            "start_seconds_in_source": float(event["start_seconds_in_source"]),
            "end_seconds_in_source": float(event["end_seconds_in_source"]),
        }
        for event in events
    ])


def validate_completed_stages(manifest: dict[str, Any], video_sha256: str) -> list[dict[str, Any]]:
    identity = manifest.get("identity", {})
    artifacts = manifest.get("artifacts", {})
    stages = artifacts.get("stage_status", {})
    if (
        manifest.get("video_sha256") != video_sha256
        or identity.get("video_sha256") != video_sha256
        or identity.get("model") != MODEL
    ):
        raise G513ValidationError("BASELINE_IDENTITY_MISMATCH")
    for stage in ("V1", "V2", "V3"):
        if stages.get(stage, {}).get("status") not in {"COMPLETED", "COMPLETED_REUSED"}:
            raise G513ValidationError(f"BASELINE_{stage}_NOT_COMPLETED")
    provenance = manifest.get("provenance", [])
    if any(
        item.get("stage") in {"v1_discovery", "v2_classification", "v3_segmentation"}
        and item.get("model") != MODEL
        for item in provenance if isinstance(item, dict)
    ):
        raise G513ValidationError("BASELINE_PROVENANCE_MODEL_MISMATCH")
    events = artifacts.get("checkpoints", {}).get("temporal_events", {}).get("events")
    if not isinstance(events, list) or not events:
        raise G513ValidationError("BASELINE_V3_EVENTS_MISSING")
    ids = set()
    for event in events:
        if (
            not isinstance(event, dict)
            or not isinstance(event.get("event_id"), str)
            or event["event_id"] in ids
            or not isinstance(event.get("candidate_id"), str)
            or float(event.get("end_seconds_in_source", 0)) <= float(event.get("start_seconds_in_source", 0))
        ):
            raise G513ValidationError("BASELINE_V3_IDS_OR_TIMING_INVALID")
        ids.add(event["event_id"])
    return events

def verify_legacy_id_mapping(events, replay):
    expected, mapping = {e["event_id"] for e in events}, {}
    for item in replay.get("accepted_events", []):
        legacy_id = item.get("event_id") if isinstance(item, dict) else None
        if not isinstance(legacy_id, str) or ":" not in legacy_id: raise G513ValidationError("V5_LEGACY_ID_INVALID")
        if legacy_id in expected:
            mapping[legacy_id] = legacy_id
            continue
        candidate, index = legacy_id.rsplit(":", 1)
        if not index.isdigit(): raise G513ValidationError("V5_LEGACY_ID_MAPPING_MISMATCH")
        event_id = f"{candidate}:e_{int(index)}"
        if event_id not in expected or legacy_id in mapping: raise G513ValidationError("V5_LEGACY_ID_MAPPING_MISMATCH")
        if not isinstance(item.get("text_verbatim"), str) or not item["text_verbatim"].strip(): raise G513ValidationError("V5_SOURCE_TEXT_EMPTY")
        mapping[legacy_id] = event_id
    if set(mapping.values()) != expected: raise G513ValidationError("V5_LEGACY_ID_MAPPING_INCOMPLETE")
    return mapping


def source_text_by_event(events, replay):
    mapping = verify_legacy_id_mapping(events, replay)
    return {mapping[x["event_id"]]: x["text_verbatim"].strip() for x in replay["accepted_events"]}


def _valid_provenance(records, stage, translation=False):
    records = [x for x in records if isinstance(x, dict) and x.get("stage") == stage] if isinstance(records, list) else []
    fields = ("schema_version", "validator_version")
    return bool(records) and all(x.get("model") == MODEL and x.get("provider") == PROVIDER and all(isinstance(x.get(k), str) and x[k] for k in fields) for x in records)


def validate_v5_replay(events, replay, video_sha256):
    accepted, expected = replay.get("accepted_events"), {e["event_id"]: e for e in events}
    if (replay.get("video_sha256") != video_sha256 or len(expected) != 15 or not isinstance(accepted, list) or len(accepted) != 15 or replay.get("v3_event_timing_sha256") != event_timing_hash(events) or not _valid_provenance(replay.get("provenance"), "v5_transcription")):
        raise G513ValidationError("V5_CHECKPOINT_IDENTITY_BLOCKED")
    output = {}
    for item in accepted:
        if not isinstance(item, dict) or item.get("event_id") not in expected or item["event_id"] in output or not isinstance(item.get("text_verbatim"), str) or not item["text_verbatim"].strip(): raise G513ValidationError("V5_CHECKPOINT_IDENTITY_BLOCKED")
        try: start, end = float(item["start_seconds_in_source"]), float(item["end_seconds_in_source"])
        except (KeyError, TypeError, ValueError) as exc: raise G513ValidationError("V5_CHECKPOINT_IDENTITY_BLOCKED") from exc
        source = expected[item["event_id"]]
        if abs(start - float(source["start_seconds_in_source"])) > .1 or abs(end - float(source["end_seconds_in_source"])) > .1: raise G513ValidationError("V5_CHECKPOINT_IDENTITY_BLOCKED")
        output[item["event_id"]] = item["text_verbatim"].strip()
    if set(output) != set(expected): raise G513ValidationError("V5_CHECKPOINT_IDENTITY_BLOCKED")
    return output


def validate_translation_checkpoint(checkpoint, events, v5_sha256):
    expected = {e["event_id"] for e in events}
    if checkpoint.get("input_v5_sha256") != v5_sha256 or not _valid_provenance(checkpoint.get("provenance"), "translation", True) or not isinstance(checkpoint.get("events"), list): return [], [e["event_id"] for e in events]
    seen, reused = set(), []
    for item in checkpoint["events"]:
        if not isinstance(item, dict) or item.get("event_id") not in expected or item.get("event_id") in seen or not isinstance(item.get("translated_text"), str) or not item["translated_text"].strip(): return [], [e["event_id"] for e in events]
        seen.add(item["event_id"]); reused.append({"event_id": item["event_id"], "translated_text": item["translated_text"].strip()})
    return (reused, []) if seen == expected else ([], [e["event_id"] for e in events])


def geometry_frame_times(event):
    start, end = float(event["start_seconds_in_source"]), float(event["end_seconds_in_source"]); duration = end - start
    if duration < .8: return [("middle", (start + end) / 2)]
    if duration <= 1.5: return [("early", start + duration * .15), ("late", end - duration * .15)]
    return [("early", start + duration * .15), ("middle", (start + end) / 2), ("late", end - duration * .15)]


def build_request_plan(events, source_text, *, video_sha256, batch_size, rpm=15):
    if not events or batch_size < 1 or rpm < 1 or set(source_text) != {e["event_id"] for e in events}: raise G513ValidationError("G513_PLAN_CONFIGURATION_INVALID")
    planned = []
    for e in events:
        frames = [{"frame_role": role, "timestamp_seconds": round(ts, 3)} for role, ts in geometry_frame_times(e)]
        planned.append({"event_id": e["event_id"], "candidate_id": e["candidate_id"], "timing": [e["start_seconds_in_source"], e["end_seconds_in_source"]], "source_text_sha256": hashlib.sha256(source_text[e["event_id"]].encode()).hexdigest(), "frame_mapping": frames, "frame_mapping_sha256": canonical_hash(frames), "image_part_count": len(frames), "frame_format": "image/jpeg", "frame_resolution": "original"})
    count = (len(events) + batch_size - 1) // batch_size
    return {"phase": "G.5.13", "provider": PROVIDER, "model": MODEL, "video_sha256": video_sha256, "event_count": len(events), "event_timing_sha256": event_timing_hash(events), "events": planned, "batch_size": batch_size, "generate_content_request_count": count, "count_tokens_call_count": count, "returned_token_count": None, "rpm": rpm, "minimum_request_spacing_seconds": max(5., 60 / rpm), "direct_video_v4_requests": 0, "completed_stage_request_count": 0, "image_part_count": sum(x["image_part_count"] for x in planned)}


def validate_no_completed_stage_requests(plan, provenance):
    if plan.get("completed_stage_request_count") != 0 or any(x.get("stage") in {"v1_discovery", "v2_classification", "v3_segmentation", "v4_geometry"} for x in provenance): raise G513ValidationError("UNEXPECTED_COMPLETED_STAGE_REQUEST")

def summarize_event(event, geometry):
    return {"event_id": event["event_id"], "timing": [event["start_seconds_in_source"], event["end_seconds_in_source"]], "geometry_status": "MISSING" if geometry is None else ("REVIEW_REQUIRED" if geometry.get("needs_review") else "ACCEPTED"), "replacement_safety": None if geometry is None else geometry.get("replacement_safety"), "review_reason": None if geometry is None else geometry.get("review_reason")}


def select_eligible_render_events(events, geometry, translations):
    gmap = {x["event_id"]: x for x in geometry}; tmap = {x["event_id"]: x for x in translations}
    return [{**e, **gmap[e["event_id"]], "translated_text": tmap[e["event_id"]]["translated_text"]} for e in events if gmap.get(e["event_id"]) and not gmap[e["event_id"]].get("needs_review") and isinstance(gmap[e["event_id"]].get("representative_box_2d"), list) and len(gmap[e["event_id"]]["representative_box_2d"]) == 4 and tmap.get(e["event_id"]) and tmap[e["event_id"]].get("translated_text", "").strip()]


def build_render_checkpoint(input_video, output_video, geometry, translations, ass_path, ffmpeg_exit_code):
    return {"request_status": RENDER_NOT_APPLICABLE, "input_video_sha256": file_hash(input_video), "output_video_sha256": file_hash(output_video), "geometry_sha256": canonical_hash(geometry), "translation_sha256": canonical_hash(translations), "ass_sha256": file_hash(ass_path), "ffmpeg_exit_code": ffmpeg_exit_code}


def validate_render_checkpoint(checkpoint, input_video, output_video, geometry, translations, ass_path):
    if checkpoint.get("request_status") != RENDER_NOT_APPLICABLE or checkpoint.get("ffmpeg_exit_code") != 0 or not output_video.is_file(): return False
    expected = build_render_checkpoint(input_video, output_video, geometry, translations, ass_path, 0)
    return all(checkpoint.get(k) == v for k, v in expected.items())


def validate_visual_qa_result(result):
    status, leakage = result.get("status"), result.get("source_text_visible")
    if status not in VISUAL_QA_STATUSES or not isinstance(leakage, bool): return {"status": "REVIEW_REQUIRED", "reason": "VISUAL_QA_UNCLEAR"}
    if leakage: return {"status": "FAIL", "reason": "SOURCE_LEAKAGE"}
    if status == "FAIL": return {"status": "FAIL", "reason": str(result.get("reason") or "VISUAL_QA_FAIL")}
    if status == "REVIEW_REQUIRED": return {"status": "REVIEW_REQUIRED", "reason": str(result.get("reason") or "VISUAL_QA_UNCLEAR")}
    return {"status": "PASS", "reason": None}


def _extract_frame(video, timestamp):
    result = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{timestamp:.3f}", "-i", str(video), "-frames:v", "1", "-f", "image2pipe", "-c:v", "mjpeg", "-"], capture_output=True, check=False)
    if result.returncode or not result.stdout: raise G513ValidationError("G513_FFMPEG_FRAME_EXTRACTION_FAILED")
    return result.stdout


def _batches(items, size):
    for idx in range(0, len(items), size): yield items[idx:idx + size]


def _artifact_path(output_root):
    return output_root / "g513_full_sample2.json"


def _pixel_box(box, width, height):
    ymin, xmin, ymax, xmax = box; return {"left": xmin * width / 1000, "top": ymin * height / 1000, "right": xmax * width / 1000, "bottom": ymax * height / 1000}


def _ass_time(seconds):
    h, rem = divmod(round(seconds * 100), 360000); m, rem = divmod(rem, 6000); return f"{h}:{m:02}:{rem // 100:02}.{rem % 100:02}"


def _escape_ass(text):
    return text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def generate_ass_file(eligible, width, height, ass_path):
    header = f"[Script Info]\nScriptType: v4.00+\nPlayResX: {width}\nPlayResY: {height}\n\n[V4+ Styles]\nFormat: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\nStyle: Default,Arial,24,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,5,0,0,0,1\n\n[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
    lines = [header]
    for event in eligible:
        box = _pixel_box(event["representative_box_2d"], width, height); l, t = round(box["left"]), round(box["top"]); r, b = round(box["right"]), round(box["bottom"]); w, h = r - l, b - t
        start, end = _ass_time(event["start_seconds_in_source"]), _ass_time(event["end_seconds_in_source"])
        fs = max(12, min(24, int(h * 0.55)))
        lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{{\\an7\\pos({l},{t})\\p1\\1c&H000000&}}m 0 0 l {w} 0 l {w} {h} l 0 {h}{{\\p0}}")
        lines.append(f"Dialogue: 1,{start},{end},Default,,0,0,0,,{{\\an5\\fs{fs}\\pos({round((l + r) / 2)},{round((t + b) / 2)})}}" + _escape_ass(event["translated_text"]))
    ass_path.parent.mkdir(parents=True, exist_ok=True); ass_path.write_text("\n".join(lines), encoding="utf-8")


def load_checkpoint(path):
    if path.is_file():
        try: return json.loads(path.read_text(encoding="utf-8"))
        except Exception: pass
    return {"v4_batches": {}, "translations": {}, "render": {}, "visual_qa": {}, "current_run_429": 0, "historical_429": 0}


def save_checkpoint(path, data):
    tmp = path.with_suffix(".tmp"); path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"); tmp.replace(path)


def validate_visual_qa_batch(results):
    for item in results:
        if item.get("source_text_visible") is True or item.get("has_duplicate") is True or item.get("has_overflow") is True or item.get("has_damage") is True: return "FAIL"
        if item.get("status") == "FAIL": return "FAIL"
    verdicts = {x.get("status") for x in results}
    return "REVIEW_REQUIRED" if "REVIEW_REQUIRED" in verdicts else ("PASS" if verdicts <= {"PASS"} else "REVIEW_REQUIRED")


def run_visual_qa_batch(provider, batch, input_video, output_video):
    parts = [{"text": "Compare source and rendered frame pairs for each event_id. Verify: 1) original source text/banner fully covered, 2) no duplicate subtitles, 3) text does not overflow, 4) text is not damaged. Return one result per event_id with source_text_visible (bool), has_duplicate (bool), has_overflow (bool), has_damage (bool), status (PASS|FAIL|REVIEW_REQUIRED), reason (string)."}]
    for event in batch:
        mid = (float(event["start_seconds_in_source"]) + float(event["end_seconds_in_source"])) / 2
        parts.extend([{"text": f"event_id={event['event_id']} source_frame"}, {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(_extract_frame(input_video, mid)).decode("ascii")}}, {"text": f"event_id={event['event_id']} rendered_frame"}, {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(_extract_frame(output_video, mid)).decode("ascii")}}])
    schema = {"type": "object", "properties": {"results": {"type": "array", "items": {"type": "object", "properties": {"event_id": {"type": "string"}, "source_text_visible": {"type": "boolean"}, "has_duplicate": {"type": "boolean"}, "has_overflow": {"type": "boolean"}, "has_damage": {"type": "boolean"}, "status": {"type": "string", "enum": ["PASS", "FAIL", "REVIEW_REQUIRED"]}, "reason": {"type": "string"}}, "required": ["event_id", "source_text_visible", "has_duplicate", "has_overflow", "has_damage", "status"]}}, "required": ["results"]}}
    request = GeminiRequest(contents=[{"parts": parts}], model=provider.model, generation_config={"responseMimeType": "application/json", "responseSchema": schema}, system_instruction="You are a quality assurance assistant.", source_kind="visual_qa", api_surface="generate_content")
    client = provider.runtime.get_client(provider.model, role=MODE_ROLE_PRIMARY, api_key=provider.api_key); request.predicted_tokens = client.count_tokens(request); response = client.generate_content(request)
    payload = parse_structured(response.text)
    if not isinstance(payload, dict): raise ProviderError("Visual QA response was not an object.")
    provider.provenance.append({"stage": "visual_qa", "input_type": "image", "model": provider.model, "count_tokens": getattr(request, "predicted_tokens", 0) or 0, "actual_usage": response.usage.to_dict() if response.usage else {}})
    return payload


def translate_batch(provider, batch, source_texts):
    items = [{"event_id": e["event_id"], "text_verbatim": source_texts[e["event_id"]]} for e in batch]
    prompt = provider._batch_translation_prompt(items, "en"); schema = provider._batch_schema("event_id", provider._translation_schema())
    request = GeminiRequest(contents=[{"parts": [{"text": prompt}]}], model=provider.model, generation_config={"responseMimeType": "application/json", "responseSchema": schema}, system_instruction=prompt, source_kind="translation", api_surface="generate_content")
    client = provider.runtime.get_client(provider.model, role=MODE_ROLE_PRIMARY, api_key=provider.api_key); request.predicted_tokens = client.count_tokens(request); response = client.generate_content(request)
    payload = parse_structured(response.text)
    if not isinstance(payload, dict): raise ProviderError("Translation response was not an object.")
    provider.provenance.append({"stage": "translation", "input_type": "text", "model": provider.model, "count_tokens": getattr(request, "predicted_tokens", 0) or 0, "actual_usage": response.usage.to_dict() if response.usage else {}})
    return payload



def build_quota_report(provenance, historical_429=0):
    calls = len(provenance)
    tokens = 0; run429 = 0
    for entry in provenance:
        usage = entry.get("actual_usage") or {}
        tokens += usage.get("candidates_token_count", usage.get("total_token_count", usage.get("totalTokenCount", 0)))
        if entry.get("status_code") == 429: run429 += 1
    return {
        "generate_content_request_count": calls,
        "count_tokens_call_count": calls,
        "returned_token_count": tokens,
        "current_run_429_count": run429,
        "historical_429_count": historical_429,
    }


def build_v4_batch_checkpoint(batch, geometry, video_hash, provenance):
    return {
        "batch_hash": canonical_hash([e["event_id"] for e in batch]),
        "video_sha256": video_hash,
        "geometry": geometry,
        "provenance": provenance,
    }


def validate_v4_batch_checkpoint(checkpoint, batch, video_hash):
    if not isinstance(checkpoint, dict) or checkpoint.get("video_sha256") != video_hash: return False
    geom_ids = {g.get("event_id") for g in checkpoint.get("geometry", []) if isinstance(g, dict)}
    batch_ids = set()
    for e in batch:
        if not isinstance(e, dict) or not isinstance(e.get("event_id"), str): return False
        batch_ids.add(e["event_id"])
    if geom_ids != batch_ids: return False
    for g in checkpoint.get("geometry", []):
        if not isinstance(g, dict) or g.get("needs_review") is True: return False
    return True

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--v5-replay", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--translation-checkpoint", type=Path, required=False)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--rpm", type=int, default=15)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--live-visual-qa", action="store_true")
    args = parser.parse_args()

    dest = _artifact_path(args.output_root)
    if dest.exists(): raise SystemExit("G513_ARTIFACT_ALREADY_EXISTS")
    video_sha256 = file_hash(args.video)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    loader = ConfigLoader(); config = loader.load()
    batch_size = args.batch_size if args.batch_size is not None else getattr(config.gemini, "direct_video_batch_size", 10)
    quota = {"generate_content_calls": 0, "count_tokens_calls": 0, "returned_token_count": 0, "current_run_429": 0, "historical_429": 0}

    try:
        events = validate_completed_stages(manifest, video_sha256)
        replay = json.loads(args.v5_replay.read_text(encoding="utf-8"))
        source_texts = validate_v5_replay(events, replay, video_sha256)
        plan = build_request_plan(events, source_texts, video_sha256=video_sha256, batch_size=batch_size, rpm=args.rpm)
        validate_no_completed_stage_requests(plan, [])
    except G513ValidationError as exc:
        artifact = {"phase": "G.5.13 Full Sample2 V4 Gemini image-level geometry", "request_status": "V5_CHECKPOINT_IDENTITY_BLOCKED" if "V5_CHECKPOINT_IDENTITY_BLOCKED" in str(exc) else "FAILED", "failure_reason": str(exc), "quota_report": quota}
        args.output_root.mkdir(parents=True, exist_ok=True); dest.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); print(f"Validation failure persisted: {exc}"); sys.exit(1)

    if args.plan_only:
        artifact = {"phase": "G.5.13 Full Sample2 V4 Gemini image-level geometry", "request_plan": plan, "baseline_manifest_sha256": file_hash(args.manifest), "v5_replay_sha256": file_hash(args.v5_replay), "v1_v3_request_status": "REUSED_NO_REQUESTS", "request_status": "PLANNED", "geometry": [], "event_summary": [summarize_event(e, None) for e in events], "visual_qa": {"status": "NOT_REQUESTED"}, "quota_report": quota}
        args.output_root.mkdir(parents=True, exist_ok=True); dest.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); print(dest); return

    checkpoint_file = args.output_root / "g513_checkpoint.json"
    checkpoint = load_checkpoint(checkpoint_file)
    quota["historical_429"] = checkpoint.get("historical_429", 0)
    provider = ProviderFactory(config=config.providers, gemini_config=config.gemini, gemini_api_key=loader.gemini_api_key).create_ocr_provider(PROVIDER)

    # V4 image-level geometry with batch resume
    geometry = []
    for batch in _batches(events, batch_size):
        bh = canonical_hash({"event_ids": [e["event_id"] for e in batch], "frame_mappings": [geometry_frame_times(e) for e in batch]})
        if bh in checkpoint.get("v4_batches", {}):
            geometry.extend(checkpoint["v4_batches"][bh])
        else:
            frame_sets = [{"event": e, "frames": [{"frame_role": r, "timestamp_seconds": ts, "image_bytes": _extract_frame(args.video, ts)} for r, ts in geometry_frame_times(e)]} for e in batch]
            try:
                quota["count_tokens_calls"] += 1
                payload = provider._request_image_geometry(provider._image_geometry_prompt(frame_sets, source_texts), provider._batch_schema("event_id", provider._image_geometry_schema()), frame_sets)
                quota["generate_content_calls"] += 1
                responses = provider._batch_responses(payload, batch, "event_id")
                bg = [provider._validate_image_geometry(responses[e["event_id"]], e) for e in batch]
                geometry.extend(bg); checkpoint.setdefault("v4_batches", {})[bh] = bg; save_checkpoint(checkpoint_file, checkpoint)
            except (GeminiRateLimitError, ProviderRateLimitError): quota["current_run_429"] += 1; raise

    # Translation with G5.9 checkpoint reuse
    v5_text_hash = canonical_hash(source_texts)
    trans_map = {}
    if args.translation_checkpoint and args.translation_checkpoint.is_file():
        raw = json.loads(args.translation_checkpoint.read_text(encoding="utf-8"))
        reused_trans, _ = validate_translation_checkpoint(raw, events, v5_text_hash)
        for x in reused_trans: trans_map[x["event_id"]] = x["translated_text"]
    for e in events:
        if not e["event_id"] in trans_map:
            pilot = source_texts.get(e["event_id"])
            if not pilot or not pilot.strip(): raise G513ValidationError("Translation: Empty pilot text is not accepted")
    rerun_events = [e for e in events if e["event_id"] not in trans_map]
    if rerun_events:
        for batch in _batches(rerun_events, batch_size):
            try:
                quota["count_tokens_calls"] += 1
                payload = translate_batch(provider, batch, source_texts)
                quota["generate_content_calls"] += 1
                for item in payload.get("results", []):
                    if item.get("event_id") and item.get("translated_text"): trans_map[item["event_id"]] = item["translated_text"].strip()
            except (GeminiRateLimitError, ProviderRateLimitError): quota["current_run_429"] += 1; raise

    # Render with actual FFmpeg
    w = manifest.get("artifacts", {}).get("video_metadata", {}).get("width", 1280); h = manifest.get("artifacts", {}).get("video_metadata", {}).get("height", 720)
    eligible = select_eligible_render_events(events, geometry, [{"event_id": k, "translated_text": v} for k, v in trans_map.items()])
    if not eligible: raise G513ValidationError("ZERO_EVENTS_RENDERED")
    output_video = args.output_root / "rendered_output.mp4"; ass_path = args.output_root / "subtitles.ass"
    render_hash_base = {"input_video_sha256": video_sha256, "geometry_sha256": canonical_hash(geometry), "translations_sha256": canonical_hash(trans_map)}
    generate_ass_file(eligible, w, h, ass_path)
    render_hash_base["ass_sha256"] = file_hash(ass_path)
    expected_render_hash = canonical_hash(render_hash_base)
    if not (checkpoint.get("render", {}).get("hash") == expected_render_hash and output_video.is_file()):
        cmd = ["ffmpeg", "-y", "-i", str(args.video), "-vf", f"ass={ass_path}", str(output_video)]
        res = subprocess.run(cmd, capture_output=True)
        if res.returncode != 0: raise G513ValidationError(f"FFmpeg rendering failed: {res.stderr.decode('utf-8', errors='ignore')}")
        checkpoint["render"] = {"hash": expected_render_hash, "output_video_sha256": file_hash(output_video)}; save_checkpoint(checkpoint_file, checkpoint)

    # Visual QA
    vqa_status = "NOT_REQUESTED"; vqa_results = []
    if args.live_visual_qa:
        vqa_hash = canonical_hash({"output_video_sha256": file_hash(output_video), "geometry_sha256": canonical_hash(geometry), "translations_sha256": canonical_hash(trans_map)})
        if checkpoint.get("visual_qa", {}).get("hash") == vqa_hash:
            vqa_results = checkpoint["visual_qa"].get("results", []); vqa_status = checkpoint["visual_qa"].get("status", "PASS")
        else:
            for batch in _batches(eligible, batch_size):
                try:
                    quota["count_tokens_calls"] += 1
                    payload = run_visual_qa_batch(provider, batch, args.video, output_video)
                    quota["generate_content_calls"] += 1
                    vqa_results.extend(payload.get("results", []))
                except (GeminiRateLimitError, ProviderRateLimitError): quota["current_run_429"] += 1; raise
            vqa_status = validate_visual_qa_batch(vqa_results)
            checkpoint["visual_qa"] = {"hash": vqa_hash, "status": vqa_status, "results": vqa_results}; save_checkpoint(checkpoint_file, checkpoint)

    for prov in provider.provenance: quota["returned_token_count"] += prov.get("actual_usage", {}).get("candidates_token_count", 0)
    checkpoint["historical_429"] = quota["historical_429"] + quota["current_run_429"]; save_checkpoint(checkpoint_file, checkpoint)

    artifact = {"phase": "G.5.13 Full Sample2 V4 Gemini image-level geometry", "request_plan": plan, "baseline_manifest_sha256": file_hash(args.manifest), "v5_replay_sha256": file_hash(args.v5_replay), "v1_v3_request_status": "REUSED_NO_REQUESTS", "request_status": "COMPLETED", "geometry": geometry, "provenance": provider.provenance, "event_summary": [summarize_event(e, next((g for g in geometry if g["event_id"] == e["event_id"]), None)) for e in events], "visual_qa": {"status": vqa_status, "results": vqa_results if args.live_visual_qa else None}, "quota_report": quota}
    dest.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); print(dest)


if __name__ == "__main__":
    main()
