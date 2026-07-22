"""Fail-closed local behavioral rejection for uncertain text tracks."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable

from .crop_extractor import TrackCropExtractor


@dataclass(frozen=True)
class BehaviorEvidence:
    speech_overlap_ratio: float
    coverage_ratio: float
    duration_seconds: float
    displacement_px: float
    transcript_available: bool
    rule_fired: bool
    rule_reason: str
    static_text_similarity: float | None = None
    paddle_text_samples: tuple[str, ...] = ()


class TrackBehaviorDiscriminator:
    """Reject only durable, static, non-dialogue tracks before Gemini."""

    def __init__(
        self,
        speech_segments: Iterable[Any] | None = None,
        video_duration_seconds: float = 0.0,
        static_text_reader: Any | None = None,
    ) -> None:
        self._speech_segments = list(speech_segments or [])
        self._video_duration_seconds = max(float(video_duration_seconds), 0.0)
        self._static_text_reader = static_text_reader

    def evaluate(self, track: dict[str, Any]) -> BehaviorEvidence:
        groups = track.get("groups", [])
        start = float(groups[0].get("timestamp_ms", 0.0)) / 1000.0 if groups else 0.0
        end = float(groups[-1].get("timestamp_ms", 0.0)) / 1000.0 if groups else start
        duration = max(0.0, end - start)
        overlap = sum(
            max(0.0, min(end, self._end(segment)) - max(start, self._start(segment)))
            for segment in self._speech_segments
        )
        overlap_ratio = min(1.0, overlap / duration) if duration else 0.0
        coverage_ratio = (
            min(1.0, duration / self._video_duration_seconds)
            if self._video_duration_seconds
            else 0.0
        )
        displacement = self._displacement(track)
        transcript_available = bool(self._speech_segments)
        is_static_watermark = (
            transcript_available
            and duration >= 3.0
            and displacement <= 12.0
            and overlap_ratio <= 0.05
            and coverage_ratio >= 0.10
        )
        return BehaviorEvidence(
            speech_overlap_ratio=round(overlap_ratio, 4),
            coverage_ratio=round(coverage_ratio, 4),
            duration_seconds=round(duration, 4),
            displacement_px=round(displacement, 4),
            transcript_available=transcript_available,
            rule_fired=is_static_watermark,
            rule_reason=(
                "durable_static_without_speech"
                if is_static_watermark
                else "neutral"
            ),
        )

    def evaluate_static_content(
        self, track: dict[str, Any], proposal_bundle: Any
    ) -> BehaviorEvidence:
        """Reject only durable, spatially static tracks with unchanged Paddle text."""
        evidence = self.evaluate(track)
        if evidence.duration_seconds < 3.0 or evidence.displacement_px > 12.0:
            return evidence

        crops = TrackCropExtractor().extract_evidence(track, proposal_bundle).get(
            "crops", []
        )
        samples = tuple(self._read_static_text(crop) for crop in crops)
        normalized = [self._normalize_text(text) for text in samples if text.strip()]
        similarity = self._text_similarity(normalized)
        static = len(normalized) >= 2 and similarity >= 0.95
        return replace(
            evidence,
            static_text_similarity=round(similarity, 4) if normalized else None,
            paddle_text_samples=samples,
            rule_fired=evidence.rule_fired or static,
            rule_reason=(
                "durable_static_paddle_text"
                if static
                else evidence.rule_reason
            ),
        )

    def _read_static_text(self, crop: Any) -> str:
        if self._static_text_reader is not None:
            return str(self._static_text_reader(crop))
        try:
            import paddleocr

            if not hasattr(self, "_paddle_ocr"):
                self._paddle_ocr = paddleocr.PaddleOCR(
                    lang="ch",
                    device="gpu",
                    enable_mkldnn=False,
                    engine_config={"paddle_static": {"enable_new_ir": False}},
                )
            result = self._paddle_ocr.ocr(crop)
            page = result[0] if result else []
            if isinstance(page, dict):
                return " ".join(str(item) for item in page.get("rec_texts", []))
            return " ".join(str(line[1][0]) for line in page if len(line) == 2)
        except Exception:
            return ""

    @staticmethod
    def _normalize_text(text: str) -> str:
        return "".join(char.casefold() for char in text if char.isalnum())

    @staticmethod
    def _text_similarity(texts: list[str]) -> float:
        if len(texts) < 2:
            return 0.0
        from difflib import SequenceMatcher

        pairs = [
            SequenceMatcher(None, left, right).ratio()
            for index, left in enumerate(texts)
            for right in texts[index + 1 :]
        ]
        return min(pairs) if pairs else 0.0

    @staticmethod
    def _start(segment: Any) -> float:
        return float(segment.start if hasattr(segment, "start") else segment.get("start", 0.0))

    @staticmethod
    def _end(segment: Any) -> float:
        return float(segment.end if hasattr(segment, "end") else segment.get("end", 0.0))

    @staticmethod
    def _displacement(track: dict[str, Any]) -> float:
        groups = track.get("groups", [])
        if len(groups) < 2:
            return 0.0
        first = groups[0]["union_bbox_xyxy"]
        last = groups[-1]["union_bbox_xyxy"]
        first_center = ((first[0] + first[2]) / 2.0, (first[1] + first[3]) / 2.0)
        last_center = ((last[0] + last[2]) / 2.0, (last[1] + last[3]) / 2.0)
        return (
            (first_center[0] - last_center[0]) ** 2
            + (first_center[1] - last_center[1]) ** 2
        ) ** 0.5
