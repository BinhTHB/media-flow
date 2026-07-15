from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.domain.entities.frame import Frame
from src.infrastructure.factory.provider_factory import ProviderFactory
from src.infrastructure.gemini.runtime import GeminiRuntime, RuntimeConfig
from src.infrastructure.providers.ocr.gemini_vision_ocr_provider import GeminiVisionOCRProvider


class _MockTransport:
    def send(self, method, url, *, headers, data=None, timeout=60.0):
        if method == "GET":
            return {"name": "models/mock", "supportedGenerationMethods": ["generateContent"]}
        request = json.loads(data.decode("utf-8"))
        frames = []
        for part in request["contents"][0]["parts"]:
            text = part.get("text", "")
            if text.startswith("frame_id="):
                frame_id, timestamp = text.replace("frame_id=", "").split("; timestamp=")
                frames.append({"frame_id": frame_id, "timestamp": float(timestamp), "detections": []})
        return {"candidates": [{"content": {"parts": [{"text": json.dumps({"frames": frames})}]}}]}


def _frames(directory: Path) -> list[Frame]:
    images = sorted(path for path in directory.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise ValueError(f"No JPG or PNG frames found in {directory}")
    return [Frame(timestamp=float(index), image_path=str(path)) for index, path in enumerate(images)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Gemini OCR qualification shadow mode on frame artifacts.")
    parser.add_argument("--frames-dir", required=True, type=Path)
    parser.add_argument("--job-id", default="gemini_ocr_qualification")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--mock", action="store_true", help="Use deterministic offline Gemini responses.")
    parser.add_argument("--dry-run", action="store_true", help="Validate frame discovery and artifact layout without a live request.")
    args = parser.parse_args()

    frames = _frames(args.frames_dir)
    if args.dry_run:
        provider = GeminiVisionOCRProvider()
        provider.write_qualification_artifacts(args.job_id, frames, [], output_dir=args.output_dir)
        return 0
    if args.mock:
        runtime = GeminiRuntime.create(RuntimeConfig(api_key="mock-key"))
        provider = GeminiVisionOCRProvider(runtime=runtime, api_key="mock-key")
        runtime.get_client(provider.model, role="primary")._transport = _MockTransport()
    else:
        if not os.environ.get("GEMINI_API_KEY"):
            parser.error("GEMINI_API_KEY is required unless --mock or --dry-run is used.")
        config = {"defaults": {"ocr": "gemini_vision"}, "providers": {"ocr": {"gemini_vision": {}}}}
        provider = ProviderFactory(config=config).create_ocr_provider("gemini_vision")
    segments = provider.detect_text(frames)
    provider.write_qualification_artifacts(args.job_id, frames, segments, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
