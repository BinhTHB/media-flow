"""Pure, offline metrics and validation for the Phase E comparison."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

FAILURE_TAXONOMY = (
    "INPUT_INVALID",
    "PROVIDER_CONFIGURATION_ERROR",
    "MODEL_UNAVAILABLE",
    "QUOTA_EXHAUSTED",
    "TRANSIENT_API_ERROR",
    "STRUCTURED_OUTPUT_INVALID",
    "SEMANTIC_VALIDATION_FAILED",
    "REFERENCE_MISSING",
    "BENCHMARK_INCOMPLETE",
    "QUALITY_FAILURE",
)
KEY_PHRASES = ("因果", "英国")


class BenchmarkValidationError(ValueError):
    """Invalid benchmark corpus input with a stable failure category."""

    failure_code = "INPUT_INVALID"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    return unicodedata.normalize("NFC", str(value).replace("\r\n", "\n").replace("\r", "\n")).strip()


def levenshtein_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for index, left_char in enumerate(left, 1):
        current = [index]
        for right_index, right_char in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + (left_char != right_char),
            ))
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    reference = normalize_text(reference)
    hypothesis = normalize_text(hypothesis)
    return levenshtein_distance(reference, hypothesis) / max(len(reference), 1)


def word_error_rate(reference: str, hypothesis: str) -> float:
    reference_words, hypothesis_words = normalize_text(reference).split(), normalize_text(hypothesis).split()
    denominator = max(len(reference_words), 1)
    if len(reference_words) < len(hypothesis_words):
        reference_words, hypothesis_words = hypothesis_words, reference_words
    previous = list(range(len(hypothesis_words) + 1))
    for index, word in enumerate(reference_words, 1):
        current = [index]
        for other_index, other_word in enumerate(hypothesis_words, 1):
            current.append(min(current[-1] + 1, previous[other_index] + 1, previous[other_index - 1] + (word != other_word)))
        previous = current
    return previous[-1] / denominator


def _box(box: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        ymin, xmin, ymax, xmax = (float(value) for value in box)
    except (TypeError, ValueError):
        return None
    return (ymin, xmin, ymax, xmax) if ymax > ymin and xmax > xmin else None


def bbox_iou(left: Any, right: Any) -> float:
    left_box, right_box = _box(left), _box(right)
    if left_box is None or right_box is None:
        return 0.0
    top, bottom = max(left_box[0], right_box[0]), min(left_box[2], right_box[2])
    start, end = max(left_box[1], right_box[1]), min(left_box[3], right_box[3])
    intersection = max(0.0, bottom - top) * max(0.0, end - start)
    left_area = (left_box[2] - left_box[0]) * (left_box[3] - left_box[1])
    right_area = (right_box[2] - right_box[0]) * (right_box[3] - right_box[1])
    return intersection / (left_area + right_area - intersection) if intersection else 0.0


def bbox_center_error(left: Any, right: Any) -> float:
    left_box, right_box = _box(left), _box(right)
    if left_box is None or right_box is None:
        return float("inf")
    left_y, left_x = (left_box[0] + left_box[2]) / 2, (left_box[1] + left_box[3]) / 2
    right_y, right_x = (right_box[0] + right_box[2]) / 2, (right_box[1] + right_box[3]) / 2
    return ((left_y - right_y) ** 2 + (left_x - right_x) ** 2) ** 0.5


def timestamp_mae(reference: Iterable[dict[str, Any]], hypothesis: Iterable[dict[str, Any]]) -> float:
    errors: list[float] = []
    for expected, actual in zip(reference, hypothesis):
        for key in ("start", "end"):
            if key in expected and key in actual:
                errors.append(abs(float(expected[key]) - float(actual[key])))
    return sum(errors) / len(errors) if errors else 0.0


def _items(value: Any, key: str) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        value = value.get(key, value.get("detections" if key == "ocr" else "segments", []))
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _text(item: dict[str, Any]) -> str:
    return str(item.get("text_verbatim", item.get("text", "")))


def _bbox(item: dict[str, Any]) -> Any:
    return item.get("box_2d", item.get("bbox", item.get("box")))


def aggregate_ocr_metrics(reference: Any, hypothesis: Any, iou_threshold: float = 0.5) -> dict[str, Any]:
    expected, actual = _items(reference, "ocr"), _items(hypothesis, "ocr")
    remaining = set(range(len(actual)))
    matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for item in expected:
        candidate = max(remaining, key=lambda index: bbox_iou(_bbox(item), _bbox(actual[index])), default=None)
        if candidate is not None and bbox_iou(_bbox(item), _bbox(actual[candidate])) >= iou_threshold:
            remaining.remove(candidate)
            matches.append((item, actual[candidate]))
    true_positive, false_positive = len(matches), len(remaining)
    misses = len(expected) - true_positive
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / len(expected) if expected else 0.0
    exact = sum(_text(left) == _text(right) for left, right in matches)
    normalized = sum(normalize_text(_text(left)) == normalize_text(_text(right)) for left, right in matches)
    return {
        "reference_count": len(expected), "prediction_count": len(actual),
        "true_positives": true_positive, "false_positives": false_positive, "misses": misses,
        "precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "exact_text_count": exact, "exact_text_rate": exact / true_positive if true_positive else 0.0,
        "normalized_text_count": normalized, "normalized_text_rate": normalized / true_positive if true_positive else 0.0,
        "cer": sum(character_error_rate(_text(left), _text(right)) for left, right in matches) / len(matches) if matches else 0.0,
        "bbox_iou": sum(bbox_iou(_bbox(left), _bbox(right)) for left, right in matches) / len(matches) if matches else 0.0,
        "bbox_center_error": sum(bbox_center_error(_bbox(left), _bbox(right)) for left, right in matches) / len(matches) if matches else 0.0,
    }


def aggregate_asr_metrics(reference: Any, hypothesis: Any) -> dict[str, Any]:
    expected, actual = _items(reference, "asr"), _items(hypothesis, "asr")
    reference_text, actual_text = " ".join(_text(item) for item in expected), " ".join(_text(item) for item in actual)
    reference_duration = max((float(item.get("end", 0)) for item in expected), default=0.0)
    covered = sum(max(0.0, min(float(item.get("end", 0)), reference_duration) - max(0.0, float(item.get("start", 0)))) for item in actual)
    return {
        "reference_segment_count": len(expected), "segment_count": len(actual),
        "cer": character_error_rate(reference_text, actual_text), "wer": word_error_rate(reference_text, actual_text),
        "coverage": min(covered / reference_duration, 1.0) if reference_duration else 0.0,
        "timestamp_mae": timestamp_mae(expected, actual),
        "exact_key_phrase_counts": {phrase: actual_text.count(phrase) for phrase in KEY_PHRASES},
        "key_phrase_counts": {phrase: actual_text.count(phrase) for phrase in KEY_PHRASES},
        "reference_key_phrase_counts": {phrase: reference_text.count(phrase) for phrase in KEY_PHRASES},
    }


def validate_manifest(manifest: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(manifest, (str, Path)):
        path = Path(manifest).resolve()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise BenchmarkValidationError(f"Cannot read manifest: {error}") from error
        root = path.parent
    else:
        payload, root = deepcopy(manifest), Path.cwd()
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise BenchmarkValidationError("Manifest must contain a samples list.")
    identifiers: set[str] = set()
    for sample in payload["samples"]:
        if not isinstance(sample, dict) or not isinstance(sample.get("id"), str) or not sample["id"] or sample["id"] in identifiers:
            raise BenchmarkValidationError("Each sample requires a unique id.")
        identifiers.add(sample["id"])
        frames, audio = sample.get("frames"), sample.get("audio")
        if not isinstance(frames, list) or not isinstance(audio, dict):
            raise BenchmarkValidationError(f"Sample {sample['id']} requires frames and audio.")
        for input_item in [*frames, audio]:
            if not isinstance(input_item, dict) or not isinstance(input_item.get("path"), str) or not isinstance(input_item.get("sha256"), str):
                raise BenchmarkValidationError(f"Sample {sample['id']} input requires path and sha256.")
            resolved = (root / input_item["path"]).resolve() if not Path(input_item["path"]).is_absolute() else Path(input_item["path"])
            if not resolved.is_file() or sha256_file(resolved).lower() != input_item["sha256"].lower():
                raise BenchmarkValidationError(f"Sample {sample['id']} input hash mismatch: {input_item['path']}")
            input_item["path"] = str(resolved)
        if not isinstance(sample.get("references", {}), dict):
            raise BenchmarkValidationError(f"Sample {sample['id']} references must be an object.")
        sample["reference_status"] = "OK" if sample["references"].get("ocr") is not None and sample["references"].get("asr") is not None else "REFERENCE_MISSING"
    return payload


def evaluate_safety_regression(manifest: dict[str, Any]) -> list[dict[str, str]]:
    """Assert the named hard safety case if it is present in corpus metadata."""
    failures: list[dict[str, str]] = []
    for sample in manifest.get("samples", []):
        metadata = sample.get("metadata", {})
        intervals = metadata.get("intervals", {}) if isinstance(metadata, dict) else {}
        interval = intervals.get("interval_6") if isinstance(intervals, dict) else None
        if sample.get("id") == "sample5" and interval is not None and (
            not isinstance(interval, dict) or
            interval.get("status") != "SKIPPED" or interval.get("reason") != "true_person_overlap_risk"
        ):
            failures.append({"sample_id": "sample5", "code": "QUALITY_FAILURE", "message": "sample5 interval_6 safety regression"})
    return failures


def sanitize_artifact(value: Any) -> Any:
    if isinstance(value, list):
        return [sanitize_artifact(item) for item in value]
    if not isinstance(value, dict):
        return value
    sanitized: dict[str, Any] = {}
    for key, item in value.items():
        lowered = key.lower().replace("-", "_")
        media_data = lowered in {"inline_data", "base64"} or (
            lowered == "data" and isinstance(item, str) and len(item) > 128 and re.fullmatch(r"[A-Za-z0-9+/=\s]+", item)
        )
        secret = lowered == "authorization" or "api_key" in lowered or lowered.endswith("apikey")
        if media_data or secret:
            sanitized[key] = "[REDACTED]"
        else:
            sanitized[key] = sanitize_artifact(item)
    return sanitized


def classify_failure(value: Any) -> str:
    text = str(value).upper()
    if any(code in text for code in FAILURE_TAXONOMY):
        return next(code for code in FAILURE_TAXONOMY if code in text)
    if "429" in text or "QUOTA" in text:
        return "QUOTA_EXHAUSTED"
    if any(token in text for token in ("TIMEOUT", "CONNECTION", "TEMPORARY", "503")):
        return "TRANSIENT_API_ERROR"
    if "MODEL" in text and any(token in text for token in ("UNAVAILABLE", "NOT FOUND", "404")):
        return "MODEL_UNAVAILABLE"
    if any(token in text for token in ("API KEY", "AUTH", "CONFIG")):
        return "PROVIDER_CONFIGURATION_ERROR"
    if any(token in text for token in ("JSON", "SCHEMA", "STRUCTURED")):
        return "STRUCTURED_OUTPUT_INVALID"
    return "SEMANTIC_VALIDATION_FAILED"
