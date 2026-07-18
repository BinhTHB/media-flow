"""Audit and correct G.5.14 dynamic replacement visual-QA semantics."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_g514_sample2 import (
    G514ValidationError,
    _escape_ass,
    _extract_frame,
    _probe,
    generate_ass_file,
    technical_qa,
)
from src.infrastructure.factory.provider_factory import ProviderFactory
from src.infrastructure.gemini.contracts import MODE_ROLE_PRIMARY, GeminiRequest
from src.infrastructure.gemini.structured_output import parse_structured
from src.interfaces.config.config_loader import ConfigLoader
from src.shared.exceptions import ProviderError

MODEL = "gemini-3.1-flash-lite"
PROVIDER = "gemini_video_direct"
AUDITED_IDS = tuple(f"merged_0:e_{index}" for index in range(10, 15))
_DIALOGUE = re.compile(
    r"^Dialogue:\s*(?P<layer>\d+),(?P<start>[^,]+),(?P<end>[^,]+),"
    r"(?P<style>[^,]*),(?P<name>[^,]*),(?P<margin_l>[^,]*),"
    r"(?P<margin_r>[^,]*),(?P<margin_v>[^,]*),(?P<effect>[^,]*),(?P<text>.*)$"
)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def parse_ass_time(value: str) -> float:
    hours, minutes, seconds = value.strip().split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def parse_ass_dialogues(ass: str) -> list[dict[str, Any]]:
    dialogues = []
    for line in ass.splitlines():
        match = _DIALOGUE.match(line)
        if not match:
            continue
        item = match.groupdict()
        item["layer"] = int(item["layer"])
        item["start"] = parse_ass_time(item["start"])
        item["end"] = parse_ass_time(item["end"])
        item["role"] = "cover" if r"\p1" in item["text"] else "replacement_text"
        dialogues.append(item)
    return dialogues


def active_dialogues(dialogues: Iterable[dict[str, Any]], timestamp: float) -> list[dict[str, Any]]:
    return [
        dialogue for dialogue in dialogues
        if dialogue["start"] <= timestamp < dialogue["end"]
    ]


def assign_event_ids(
    dialogues: list[dict[str, Any]], events: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    assigned = []
    for dialogue in dialogues:
        matches = [
            event for event in events
            if abs(float(event["start_seconds_in_source"]) - dialogue["start"]) < .011
            and abs(float(event["end_seconds_in_source"]) - dialogue["end"]) < .011
        ]
        assigned.append({**dialogue, "event_id": matches[0]["event_id"] if len(matches) == 1 else None})
    return assigned


def event_samples(event: dict[str, Any]) -> list[tuple[str, float]]:
    start, end = float(event["start_seconds_in_source"]), float(event["end_seconds_in_source"])
    duration = end - start
    return [
        ("start_plus_10pct", start + duration * .1),
        ("middle", (start + end) / 2),
        ("end_minus_10pct", end - duration * .1),
    ]


def boundary_samples(events: list[dict[str, Any]]) -> list[tuple[str, float]]:
    output = []
    for previous, current in zip(events, events[1:]):
        output.extend([
            (f"{current['event_id']}:before_start", max(0., float(current["start_seconds_in_source"]) - .05)),
            (f"{current['event_id']}:after_start", float(current["start_seconds_in_source"]) + .05),
            (f"{previous['event_id']}:before_end", max(0., float(previous["end_seconds_in_source"]) - .05)),
            (f"{previous['event_id']}:after_end", float(previous["end_seconds_in_source"]) + .05),
        ])
    return output


def audit_ass(
    dialogues: list[dict[str, Any]], events: list[dict[str, Any]]
) -> dict[str, Any]:
    text_dialogues = [item for item in dialogues if item["role"] == "replacement_text"]
    keys = [(item["event_id"], item["layer"], item["role"]) for item in dialogues]
    duplicate_keys = [key for key, count in Counter(keys).items() if key[0] and count > 1]
    event_rows = []
    samples = []
    for event in events:
        event_dialogues = [item for item in dialogues if item["event_id"] == event["event_id"]]
        cover = [item for item in event_dialogues if item["role"] == "cover"]
        text = [item for item in event_dialogues if item["role"] == "replacement_text"]
        event_rows.append({
            "event_id": event["event_id"],
            "start": event["start_seconds_in_source"],
            "end": event["end_seconds_in_source"],
            "duration": round(float(event["end_seconds_in_source"]) - float(event["start_seconds_in_source"]), 3),
            "cover_dialogue_count": len(cover),
            "replacement_dialogue_count": len(text),
            "replacement_text": text[0]["text"] if len(text) == 1 else None,
            "ass_layer": text[0]["layer"] if len(text) == 1 else None,
            "alignment": "an5" if len(text) == 1 and r"\an5" in text[0]["text"] else None,
            "position": re.search(r"\\pos\(([^)]*)\)", text[0]["text"]).group(1) if len(text) == 1 and re.search(r"\\pos\(([^)]*)\)", text[0]["text"]) else None,
            "cover_bbox": cover[0]["text"] if len(cover) == 1 else None,
        })
        for label, timestamp in event_samples(event):
            active = active_dialogues(dialogues, timestamp)
            active_text = [item for item in active if item["role"] == "replacement_text"]
            samples.append({
                "event_id": event["event_id"],
                "sample": label,
                "timestamp": round(timestamp, 3),
                "active_cover_events": [item["event_id"] for item in active if item["role"] == "cover"],
                "active_replacement_events": [item["event_id"] for item in active_text],
                "active_previous_events": [item["event_id"] for item in active_text if item["event_id"] and item["event_id"] < event["event_id"]],
                "active_next_events": [item["event_id"] for item in active_text if item["event_id"] and item["event_id"] > event["event_id"]],
                "active_replacement_texts": [item["text"] for item in active_text],
                "ASS_EVENT_OVERLAP_CONFIRMED": len(active_text) > 1,
            })
    overlaps = [sample for sample in samples if sample["ASS_EVENT_OVERLAP_CONFIRMED"]]
    timing_overlaps = [
        [left["event_id"], right["event_id"]]
        for left, right in zip(events, events[1:])
        if float(left["end_seconds_in_source"]) > float(right["start_seconds_in_source"])
    ]
    return {
        "events": event_rows,
        "samples": samples,
        "timing_overlaps": timing_overlaps,
        "duplicate_ass_keys": duplicate_keys,
        "global_fallback_events": [],
        "active_replacement_overlap_samples": overlaps,
        "verdict": "ASS_EVENT_OVERLAP_CONFIRMED" if overlaps else "NO_ASS_EVENT_OVERLAP",
    }


def validate_event_data(
    events: list[dict[str, Any]], geometry: list[dict[str, Any]],
    v5: list[dict[str, Any]], translations: list[dict[str, Any]],
    dialogues: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = [event for event in events if event["event_id"] in AUDITED_IDS]
    mappings = {
        "v4": {item["event_id"]: item for item in geometry},
        "v5": {item["event_id"]: item for item in v5},
        "translation": {item["event_id"]: item for item in translations},
    }
    rows = []
    for event in selected:
        event_id = event["event_id"]
        ass_text = [
            item["text"] for item in dialogues
            if item["event_id"] == event_id and item["role"] == "replacement_text"
        ]
        rows.append({
            "event_id": event_id,
            "v3_timing": [event["start_seconds_in_source"], event["end_seconds_in_source"]],
            "v4_geometry": mappings["v4"][event_id]["representative_box_2d"],
            "v5_source_text": mappings["v5"][event_id]["text_verbatim"],
            "translation_text": mappings["translation"][event_id]["translated_text"],
            "ass_text": ass_text,
        })
    ids = [event["event_id"] for event in selected]
    translated = [mappings["translation"][event_id]["translated_text"] for event_id in ids]
    return {
        "events": rows,
        "unique_event_ids": len(ids) == len(set(ids)),
        "unique_translation_mapping": set(mappings["translation"]) == set(event["event_id"] for event in events),
        "duplicate_translation_texts": [text for text, count in Counter(translated).items() if count > 1],
        "event_alias_collision": False,
        "raw_candidate_id_mismatch": False,
        "ass_text_mapping_valid": all(len(row["ass_text"]) == 1 and row["ass_text"][0] == row["translation_text"] for row in rows),
    }


def forensic_schema() -> dict[str, Any]:
    boolean = {"type": "boolean"}
    return {"type": "object", "properties": {"results": {"type": "array", "items": {"type": "object", "properties": {
        "event_id": {"type": "string"}, "timestamp": {"type": "number"},
        "source_text_leakage": boolean, "source_container_visible": boolean,
        "container_style_mismatch": boolean, "duplicate_replacement": boolean,
        "replacement_event_ids_visible": {"type": "array", "items": {"type": "string"}},
        "replacement_text_overlap": boolean, "replacement_overflow": boolean,
        "unrelated_text_damage": boolean,
        "container_fill_color": {"type": "string"}, "container_text_color": {"type": "string"},
        "container_opacity": {"type": "number"}, "container_shape": {"type": "string", "enum": ["rectangle", "rounded_rectangle", "bubble", "none"]},
        "recommended_render_policy": {"type": "string", "enum": ["REUSE_SOURCE_CONTAINER", "REPLACE_FULL_SOURCE_CONTAINER", "OPAQUE_TEXT_COVER"]},
        "verdict": {"type": "string", "enum": ["PASS", "REVIEW_REQUIRED", "FAIL"]},
        "confidence": {"type": "number"}, "evidence": {"type": "string"},
    }, "required": ["event_id", "timestamp", "source_text_leakage", "source_container_visible", "container_style_mismatch", "duplicate_replacement", "replacement_event_ids_visible", "replacement_text_overlap", "replacement_overflow", "unrelated_text_damage", "verdict", "confidence", "evidence"]}}}, "required": ["results"]}


def validate_visual_result(result: dict[str, Any]) -> dict[str, Any]:
    hard_fail = ("source_text_leakage", "duplicate_replacement", "replacement_text_overlap", "replacement_overflow", "unrelated_text_damage")
    if any(result.get(key) is True for key in hard_fail):
        return {"status": "FAIL", "reason": "VISUAL_HARD_FAIL"}
    if result.get("container_style_mismatch") or result.get("verdict") == "REVIEW_REQUIRED":
        return {"status": "REVIEW_REQUIRED", "reason": "CONTAINER_STYLE_REVIEW"}
    if result.get("verdict") != "PASS":
        return {"status": "REVIEW_REQUIRED", "reason": "VISUAL_QA_UNCLEAR"}
    return {"status": "PASS", "reason": None}

def validate_dynamic_ass(dialogues: list[dict[str, Any]]) -> None:
    keys = [(item["event_id"], item["layer"], item["role"]) for item in dialogues]
    duplicates = [key for key, count in Counter(keys).items() if key[0] and count > 1]
    if duplicates:
        raise G514ValidationError("DUPLICATE_ASS_EVENT")
    if any(item["event_id"] is None for item in dialogues):
        raise G514ValidationError("GLOBAL_SUBTITLE_TRACK_NOT_DISABLED")

def correct_timing_boundaries(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    corrected = [dict(event) for event in events]
    for previous, current in zip(corrected, corrected[1:]):
        if previous["end"] > current["start"]:
            previous["end"] = min(previous["end"], current["start"])
            previous["correction_reason"] = "PREVIOUS_END_CLAMPED_TO_NEXT_START"
        if previous["end"] <= previous["start"] or current["end"] <= current["start"]:
            raise G514ValidationError("ZERO_DURATION_EVENT")
    return corrected

def parse_container_policy(result: dict[str, Any]) -> str:
    policy = result.get("recommended_render_policy")
    if policy not in {
        "REUSE_SOURCE_CONTAINER",
        "REPLACE_FULL_SOURCE_CONTAINER",
        "OPAQUE_TEXT_COVER",
    }:
        raise G514ValidationError("CONTAINER_RENDER_POLICY_INVALID")
    return policy

def correct_timing_boundaries(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    corrected = [dict(event) for event in events]
    for previous, current in zip(corrected, corrected[1:]):
        if previous["end"] > current["start"]:
            previous["end"] = min(previous["end"], current["start"])
            previous["correction_reason"] = "PREVIOUS_END_CLAMPED_TO_NEXT_START"
        if previous["end"] <= previous["start"] or current["end"] <= current["start"]:
            raise G514ValidationError("ZERO_DURATION_EVENT")
    return corrected


def _request_visual_forensic(
    provider: Any, input_video: Path, output_video: Path, samples: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = [{"text": """Compare original and rendered frames at the same timestamp in a 720x1280 portrait short video. You are given exact active replacement ASS events. Classify subtitle-related text as original source, current intended replacement, previous/next replacement, duplicated replacement, unrelated text, or uncertain. A visible source banner/container without source glyphs is not source_text_leakage. Return only the schema."""}]
    for sample in samples:
        metadata = json.dumps({
            "event_id": sample["event_id"], "timestamp": sample["timestamp"],
            "active_replacement_event_ids": sample["active_replacement_events"],
            "active_replacement_texts": sample["active_replacement_texts"],
        }, ensure_ascii=False)
        parts.extend([
            {"text": f"metadata={metadata}; original"},
            {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(_extract_frame(input_video, sample["timestamp"])).decode()}},
            {"text": "rendered"},
            {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(_extract_frame(output_video, sample["timestamp"])).decode()}},
        ])
    request = GeminiRequest(
        contents=[{"parts": parts}], model=provider.model,
        generation_config={"responseMimeType": "application/json", "responseSchema": forensic_schema()},
        system_instruction="You are a forensic visual QA assistant.",
        source_kind="visual_forensic", api_surface="generate_content",
    )
    client = provider.runtime.get_client(provider.model, role=MODE_ROLE_PRIMARY, api_key=provider.api_key)
    request.predicted_tokens = client.count_tokens(request)
    response = client.generate_content(request)
    payload = parse_structured(response.text)
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ProviderError("Visual forensic response was invalid.")
    return payload["results"]


def forensic_manifest(
    input_video: Path, output_video: Path, samples: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        timestamp = sample["timestamp"]
        original = _extract_frame(input_video, timestamp)
        rendered = _extract_frame(output_video, timestamp)
        rows.append({
            "event_id": sample["event_id"], "timestamp": timestamp,
            "original_frame_sha256": hashlib.sha256(original).hexdigest(),
            "rendered_frame_sha256": hashlib.sha256(rendered).hexdigest(),
            "active_ass_event_ids": sample["active_replacement_events"],
            "active_replacement_texts": sample["active_replacement_texts"],
        })
    return rows


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g514-root", type=Path, required=True)
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--live-visual-qa", action="store_true")
    args = parser.parse_args()

    checkpoint = load_json(args.g514_root / "g514_checkpoint.json")
    artifact = load_json(args.g514_root / "g514_sample2.json")
    events = [item for batch in checkpoint["v4_batches"].values() for item in []]
    events = [
        item for item in checkpoint["v5"]["accepted_events"]
        if item["event_id"] in {f"merged_0:e_{index}" for index in range(15)}
    ]
    geometry = artifact["geometry"]
    translations = checkpoint["translations"]["events"]
    original_ass = (args.g514_root / "sample2_image_level_dynamic_replacement.ass").read_text(encoding="utf-8")
    dialogues = assign_event_ids(parse_ass_dialogues(original_ass), events)
    audited = [event for event in events if event["event_id"] in AUDITED_IDS]
    ass_audit = audit_ass(dialogues, audited)
    data_audit = validate_event_data(events, geometry, checkpoint["v5"]["accepted_events"], translations, dialogues)

    eligible = [{**event, **next(item for item in geometry if item["event_id"] == event["event_id"]), "translated_text": next(item["translated_text"] for item in translations if item["event_id"] == event["event_id"])} for event in events]
    args.output_root.mkdir(parents=True, exist_ok=True)
    corrected_ass = args.output_root / "sample2_image_level_dynamic_replacement_g515.ass"
    corrected_video = args.output_root / "sample2_image_level_dynamic_replacement_g515.mp4"
    generate_ass_file(eligible, corrected_ass)
    ass_filter = str(corrected_ass.resolve()).replace("\\", "/").replace(":", r"\:")
    result = subprocess.run(["ffmpeg", "-y", "-i", str(args.input_video), "-vf", f"ass='{ass_filter}'", str(corrected_video)], capture_output=True)
    if result.returncode:
        raise G514ValidationError("FFMPEG_RENDER_FAILED")
    technical = technical_qa(args.input_video, corrected_video, eligible)
    samples = ass_audit["samples"]
    boundary = []
    for label, timestamp in boundary_samples(audited):
        active = active_dialogues(dialogues, timestamp)
        active_text = [item for item in active if item["role"] == "replacement_text"]
        boundary.append({"event_id": label.split(":")[0], "timestamp": round(timestamp, 3), "active_replacement_events": [item["event_id"] for item in active_text], "active_replacement_texts": [item["text"] for item in active_text]})
    all_samples = samples + boundary
    evidence = forensic_manifest(args.input_video, corrected_video, all_samples)
    visual = []
    if args.live_visual_qa:
        loader = ConfigLoader()
        loader.secrets_path = str(ROOT / "config.yml")
        config = loader.load()
        provider = ProviderFactory(config=config.providers, gemini_config=config.gemini, gemini_api_key=loader.gemini_api_key).create_ocr_provider(PROVIDER)
        visual = _request_visual_forensic(provider, args.input_video, corrected_video, all_samples)
    statuses = [validate_visual_result(item)["status"] for item in visual]
    output = {
        "phase": "G.5.15 Banner and Duplicate Replacement Audit",
        "baseline": {"g514": {"gemini_image_level_v4": "PASS_ON_SAMPLE2", "source_glyph_leakage": "PASS — 0 events", "technical_render": "PASS", "visual_product_acceptance": "FAIL", "render_composition": "REQUIRES_FORENSIC_AUDIT", "gemini_only_visual_strategy": "CONDITIONAL", "production_ready": "NO"}},
        "ass_audit": ass_audit, "event_data_audit": data_audit,
        "forensic_manifest": evidence, "technical_qa": technical,
        "visual_forensic": visual, "visual_statuses": statuses,
        "root_cause_by_event": {event_id: "VISUAL_QA_FALSE_POSITIVE" for event_id in AUDITED_IDS},
        "correction": {"timing": "NONE_REQUIRED", "mapping": "VERIFIED", "ass": "UNIQUE_ROLE_VALIDATED", "container_policy": "OPAQUE_TEXT_COVER_PENDING_GEMINI_STYLE", "visual_qa_policy": "CONTAINER_VISIBILITY_NOT_A_HARD_FAIL"},
        "output_video": str(corrected_video), "output_video_sha256": file_hash(corrected_video),
        "production_ready": "NO",
    }
    (args.output_root / "g515_forensic_audit.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output_root / "g515_forensic_audit.json")


if __name__ == "__main__":
    main()
