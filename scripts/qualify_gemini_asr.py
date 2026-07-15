from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.infrastructure.factory.provider_factory import ProviderFactory
from src.infrastructure.gemini.runtime import GeminiRuntime, RuntimeConfig
from src.infrastructure.providers.speech.gemini_audio_speech_provider import GeminiAudioSpeechProvider

class _MockTransport:
    def send(self, method, url, *, headers, data=None, timeout=60.0):
        if method == "GET":
            return {"name": "models/mock", "supportedGenerationMethods": ["generateContent"], "modelStage": "STABLE"}
        payload = {
            "source_language": "en",
            "segments": [],
        }
        return {
            "candidates": [{"content": {"parts": [{"text": json.dumps(payload, ensure_ascii=False)}]}}],
            "modelStatus": {"modelStage": "STABLE"},
        }

def main() -> int:
    parser = argparse.ArgumentParser(description="Run Gemini ASR qualification shadow mode on an audio WAV file.")
    parser.add_argument("--audio", required=True, type=Path, help="Path to a WAV audio file for shadow qualification.")
    parser.add_argument("--job-id", default="gemini_asr_qualification")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--mock", action="store_true", help="Use deterministic offline Gemini responses (empty transcript).")
    parser.add_argument("--dry-run", action="store_true", help="Validate audio path and artifact layout without a live request.")
    args = parser.parse_args()

    audio_path = args.audio
    if not audio_path.exists():
        parser.error(f"Audio file not found: {audio_path}")

    if args.dry_run:
        provider = GeminiAudioSpeechProvider()
        provider.write_qualification_artifacts(args.job_id, [], output_dir=args.output_dir)
        return 0

    if args.mock:
        runtime = GeminiRuntime.create(RuntimeConfig(api_key="mock-key"))
        provider = GeminiAudioSpeechProvider(runtime=runtime, api_key="mock-key")
        runtime.get_client(provider.model, role="primary")._transport = _MockTransport()
        runtime.get_client(provider.escalation_model, role="escalation")._transport = _MockTransport()
    else:
        if not os.environ.get("GEMINI_API_KEY"):
            parser.error("GEMINI_API_KEY is required unless --mock or --dry-run is used.")
        config = {"defaults": {"speech": "gemini_audio"}, "providers": {"speech": {"gemini_audio": {}}}}
        provider = ProviderFactory(config=config, gemini_api_key=os.environ.get("GEMINI_API_KEY")).create_speech_provider("gemini_audio")

    segments = provider.transcribe(str(audio_path), "auto")
    provider.write_qualification_artifacts(args.job_id, segments, output_dir=args.output_dir)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
