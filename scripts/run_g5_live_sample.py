import os
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())

from src.interfaces.config.config_loader import ConfigLoader
from src.infrastructure.factory.provider_factory import ProviderFactory
from src.shared.exceptions import ProviderError

REVIEW_ROOT = Path("output/review/gemini_direct_video_g5")
REVIEW_ROOT.mkdir(parents=True, exist_ok=True)


def run_sample(sample_id: str, source_language: str = "vi"):
    print(f"\n{'='*60}\nSAMPLE {sample_id}\n{'='*60}")
    config = ConfigLoader().load()
    factory = ProviderFactory(
        config=config.providers,
        gemini_config=config.gemini,
        gemini_api_key=os.environ.get("GEMINI_API_KEY"),
    )
    provider = factory.create_ocr_provider("gemini_video_direct")
    # ensure file_manager client is initialized (provider does this internally too)
    provider.runtime.get_client(provider.model, api_key=provider.api_key)

    video_path = Path(f"test_videos/real_mini_benchmark/{sample_id}.mp4")
    out_dir = REVIEW_ROOT / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    try:
        segments = provider.extract_subtitles(video_path, source_language)
    except ProviderError as e:
        print(f"ProviderError: {e}")
        # still dump artifacts
        segments = []
    elapsed = time.monotonic() - t0

    arts = provider.debug_artifacts()
    # write artifacts (redacted)
    name_map = {
        "video_upload": "g5_video_upload.json",
        "fps_smoke_test": "g5_fps_smoke_test.json",
        "candidates": "g5_candidates.json",
        "classifications": "g5_classifications.json",
        "temporal_events": "g5_temporal_events.json",
        "geometry": "g5_geometry.json",
        "transcriptions": "g5_transcriptions.json",
        "translations": "g5_translations.json",
        "render_eligibility": "g5_render_eligibility.json",
        "final": "g5_final.json",
        "video_metadata": "g5_video_metadata.json",
    }
    for key, fname in name_map.items():
        if key in arts:
            (out_dir / fname).write_text(
                json.dumps(arts[key], ensure_ascii=False, indent=2), encoding="utf-8"
            )
    (out_dir / "g5_request_manifest.json").write_text(
        json.dumps(provider.provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "g5_usage.json").write_text(
        json.dumps(
            [{"stage": p.get("stage"), "actual_usage": p.get("actual_usage", {})} for p in provider.provenance],
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    # summary
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Segments returned: {len(segments)}")
    print(f"Provenance requests: {len(provider.provenance)}")
    for p in provider.provenance[:5]:
        print(f"  stage={p.get('stage')} input={p.get('input_type')} img_parts={p.get('image_part_count')} "
              f"local_ocr={p.get('local_ocr_calls')} fps={p.get('requested_fps')} "
              f"count={p.get('count_tokens')} usage={p.get('actual_usage')}")
    return provider, segments, arts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run G.5 direct-video live sample.")
    parser.add_argument("sample", nargs="?", default="sample2")
    parser.add_argument("--source", default="vi",
                        help="Source (original) subtitle language code, e.g. vi, zh.")
    args = parser.parse_args()
    run_sample(args.sample, source_language=args.source)
    print("\nDONE")
