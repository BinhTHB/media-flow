"""Replay persisted V5 responses without Gemini, Files API, or token calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.infrastructure.providers.ocr.gemini_video_direct_provider import (
    GeminiDirectVideoOCRProvider,
)


def replay(manifest: dict[str, Any], source_language: str = "auto") -> dict[str, Any]:
    checkpoints = manifest.get("artifacts", {}).get("checkpoints", {})
    events = checkpoints.get("temporal_events", {}).get("events", [])
    batches = checkpoints.get("v5_raw_transcription", {}).get("batches", [])
    if not isinstance(events, list) or not isinstance(batches, list):
        return {"verdict": "RAW_ARTIFACT_MISSING", "network_requests": 0,
                "count_tokens_calls": 0, "files_api_calls": 0}

    expected = {event.get("event_id"): event for event in events if isinstance(event, dict)}
    raw_results: dict[str, dict[str, Any]] = {}
    duplicate_ids: list[str] = []
    malformed = 0
    for batch in batches:
        payload = batch.get("payload") if isinstance(batch, dict) else None
        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list):
            malformed += 1
            continue
        for result in results:
            if not isinstance(result, dict) or not isinstance(result.get("event_id"), str):
                malformed += 1
                continue
            event_id = result["event_id"]
            if event_id in raw_results:
                duplicate_ids.append(event_id)
                continue
            raw_results[event_id] = result

    audits = []
    accepted = []
    for event_id, event in expected.items():
        payload = raw_results.get(event_id)
        if payload is None:
            audits.append({"event_id": event_id, "reject_reason": "MISSING_EVENT_RESULT"})
            continue
        assessment = GeminiDirectVideoOCRProvider._assess_transcription(
            payload, event, source_language
        )
        audits.append(assessment)
        if isinstance(assessment.get("transcription"), dict):
            accepted.append({"event_id": event_id, **assessment["transcription"]})

    unknown_ids = sorted(set(raw_results) - set(expected))
    artifact_hash = hashlib.sha256(
        json.dumps(batches, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    rejected = [item for item in audits if item.get("reject_reason")]
    return {
        "verdict": "V5_VALIDATION_REPLAY_PASS" if accepted else "REVIEW_REQUIRED",
        "network_requests": 0,
        "count_tokens_calls": 0,
        "files_api_calls": 0,
        "accepted_events": accepted,
        "rejected_events": rejected,
        "review_required": len(rejected),
        "raw_artifact_hash": artifact_hash,
        "validator_version": "v5_unicode_source_script_v1",
        "malformed_responses": malformed,
        "missing_event_ids": [item["event_id"] for item in audits
                              if item.get("reject_reason") == "MISSING_EVENT_RESULT"],
        "duplicate_event_ids": duplicate_ids,
        "unknown_event_ids": unknown_ids,
        "reject_counts": dict(Counter(
            item.get("reject_reason", "ACCEPTED") for item in audits
        )),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-language", default="auto")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = replay(manifest, args.source_language)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(report["verdict"])


if __name__ == "__main__":
    main()
