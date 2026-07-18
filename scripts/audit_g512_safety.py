"""Run the G.5.12 image-level safety audit for one subtitle event."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.infrastructure.factory.provider_factory import ProviderFactory
from src.interfaces.config.config_loader import ConfigLoader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--event", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    event = json.loads(args.event.read_text(encoding="utf-8"))
    loader = ConfigLoader()
    config = loader.load()
    provider = ProviderFactory(
        config=config.providers,
        gemini_config=config.gemini,
        gemini_api_key=loader.gemini_api_key,
    ).create_ocr_provider("gemini_video_direct")
    frames = []
    for frame_role, timestamp_seconds in provider._geometry_frame_times(event):
        image_bytes = provider._extract_geometry_frame(args.video, timestamp_seconds)
        output_path = args.output_root / f"{frame_role}.jpg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(image_bytes)
        frames.append({
            "frame_role": frame_role,
            "timestamp_seconds": timestamp_seconds,
            "image_bytes": image_bytes,
        })

    frame_sets = [{"event": event, "frames": frames}]
    payload = provider._request_image_geometry(
        provider._image_geometry_prompt(frame_sets),
        provider._batch_schema("event_id", provider._image_geometry_schema()),
        frame_sets,
    )
    response = provider._batch_responses(payload, [event], "event_id")[event["event_id"]]
    audit = provider._validate_image_geometry(response, event)
    artifacts = {
        "phase": "G.5.12 Gemini Image-Level Safety Semantics",
        "event": event["event_id"],
        "request_plan": {
            "events": 1,
            "image_parts": len(frames),
            "generate_requests": 1,
            "count_tokens": 1,
        },
        "request_status": "COMPLETED",
        "validation_status": "REVIEW_REQUIRED" if audit["needs_review"] else "ACCEPTED",
        "stage_status": "BLOCKED" if audit["needs_review"] else "COMPLETED",
        "geometry": audit,
        "provenance": provider.provenance,
    }
    (args.output_root / "g512_safety_audit.json").write_text(
        json.dumps(artifacts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(artifacts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
