"""Explicit cache identities for the CLI visual providers."""
from pathlib import Path
import importlib.metadata
import os

from src.infrastructure.stage_cache import (
    CachedDetector, CachedOCR, CachedTranslation, JsonStageCache, file_digest,
)


def wire_visual_caches(root, cache_dir, detector, ocr, translation, detector_config, route):
    root = Path(root)
    # Only provider code enters provider cache identities; renderer edits retain hits.
    sources = {
        "detection": ["src/infrastructure/detection/yolo.py",
                      "src/infrastructure/detection/yolo_detector.py",
                      "src/infrastructure/detection/yolo_proposal_provider.py"],
        "ocr": ["src/infrastructure/ocr/gemini_ocr.py",
                "src/infrastructure/ocr/rapid_ocr.py",
                "src/infrastructure/ocr/router.py",
                "src/infrastructure/gemini/client.py"],
        "translation": ["src/infrastructure/translation/gemini_translation.py",
                        "src/infrastructure/translation/openrouter_translation.py",
                        "src/infrastructure/translation/router.py",
                        "src/infrastructure/gemini/client.py"],
        "scene": ["src/infrastructure/video/scene_boundaries.py"],
    }
    caches = {}
    for stage, paths in sources.items():
        identity = {"code": {path: file_digest(root / path) for path in paths},
                    "cache_adapter": file_digest(root / "src/infrastructure/stage_cache.py"),
                    "wiring": file_digest(__file__)}
        if stage == "detection":
            identity.update(detector_config)
            identity["weights_sha256"] = file_digest(detector_config["weights_path"])
            identity["ultralytics"] = importlib.metadata.version("ultralytics")
        elif stage == "ocr":
            identity["model"] = getattr(ocr, "model", None)
        elif stage == "translation":
            identity["route"] = route
            identity["openrouter_models"] = os.getenv("OPENROUTER_TRANSLATION_MODELS", "")
            identity["gemini_available"] = bool(os.getenv("GEMINI_API_KEY"))
            identity["openrouter_available"] = bool(os.getenv("OPENROUTER_API_KEYS") or os.getenv("OPENROUTER_API_KEY"))
        caches[stage] = JsonStageCache(cache_dir, identity)
    if detector is not None:
        detector = CachedDetector(detector, caches["detection"])
    if ocr is not None:
        ocr = CachedOCR(ocr, caches["ocr"])
    if translation is not None:
        translation.provider = CachedTranslation(translation.provider, caches["translation"])
    return detector, ocr, translation, caches
