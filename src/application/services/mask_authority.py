"""Single authority for classifier-backed subtitle mask decisions."""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any, Mapping

from src.domain.entities.subtitle_region import SubtitleRegion


MASK_CLASSIFIER_MODELS = frozenset({"v21_pipeline"})


class DecisionStore:
    """Append-only, per-run store for classifier decisions."""

    def __init__(self, run_id: str, path: str | Path | None = None) -> None:
        self.run_id = run_id
        self.path = Path(path) if path else None
        self._records: dict[str, dict[str, Any]] = {}

    def record(
        self,
        *,
        classification_id: str,
        model: str,
        verdict: bool,
        raw_response_id: str | None,
        cache_hit: bool,
        live_or_cache: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if classification_id in self._records:
            raise ValueError(f"Duplicate mask classification_id: {classification_id}")
        record = {
            "classification_id": classification_id,
            "model": model,
            "run_id": self.run_id,
            "cache_hit": bool(cache_hit),
            "live_or_cache": live_or_cache,
            "verdict": bool(verdict),
            "raw_response_id": raw_response_id,
            **dict(metadata or {}),
        }
        self._records[classification_id] = record
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def get(self, classification_id: str | None) -> dict[str, Any] | None:
        if not classification_id:
            return None
        return self._records.get(classification_id)

    def is_positive(self, classification_id: str | None, model: str | None) -> bool:
        if model not in MASK_CLASSIFIER_MODELS:
            return False
        record = self.get(classification_id)
        return bool(
            record
            and record["run_id"] == self.run_id
            and record["model"] == model
            and record["verdict"] is True
        )

    def positive_count(self) -> int:
        return sum(record["verdict"] is True for record in self._records.values())


class MaskAuthority:
    """Issues the only provenance accepted by production ASS masking."""

    def __init__(self, store: DecisionStore) -> None:
        self.store = store

    def decide(
        self,
        candidate: Mapping[str, Any],
        decision: Mapping[str, Any],
        *,
        model: str,
        raw_response_id: str | None = None,
        cache_hit: bool = False,
    ) -> SubtitleRegion | None:
        if model not in MASK_CLASSIFIER_MODELS:
            raise ValueError(f"Mask classifier model is not whitelisted: {model}")
        verdict = decision.get("is_subtitle") is True
        classification_id = f"mask:{self.store.run_id}:{uuid.uuid4().hex}"
        self.store.record(
            classification_id=classification_id,
            model=model,
            verdict=verdict,
            raw_response_id=raw_response_id,
            cache_hit=cache_hit,
            live_or_cache="cache" if cache_hit else "live",
            metadata={"candidate_id": candidate.get("track_id")},
        )
        if not verdict:
            return None
        region = SubtitleRegion(
            start_time=float(candidate["start_time"]),
            end_time=float(candidate["end_time"]),
            x=float(candidate["x"]),
            y=float(candidate["y"]),
            width=float(candidate["width"]),
            height=float(candidate["height"]),
            confidence=float(candidate["confidence"]),
            classification_id=classification_id,
            is_subtitle=True,
            classifier_model=model,
            track_id=candidate.get("track_id"),
            proposal_ids=list(candidate.get("proposal_ids", [])),
            category=str(decision.get("category", "dialogue_subtitle")),
            classifier_confidence=decision.get("confidence"),
        )
        return region

    def decide_region(
        self,
        candidate: SubtitleRegion,
        decision: Mapping[str, Any],
        *,
        model: str,
        raw_response_id: str | None = None,
        cache_hit: bool = False,
    ) -> SubtitleRegion | None:
        """Issue provenance for an already-normalized candidate region."""
        return self.decide(
            {
                "start_time": candidate.start_time,
                "end_time": candidate.end_time,
                "x": candidate.x,
                "y": candidate.y,
                "width": candidate.width,
                "height": candidate.height,
                "confidence": candidate.confidence,
                "track_id": candidate.track_id,
                "proposal_ids": candidate.proposal_ids,
            },
            decision,
            model=model,
            raw_response_id=raw_response_id,
            cache_hit=cache_hit,
        )


def has_verified_mask_provenance(region: SubtitleRegion, store: DecisionStore) -> bool:
    """Validate the region against the current run's decision source."""
    valid = store.is_positive(region.classification_id, region.classifier_model)
    if not valid:
        logging.getLogger(__name__).warning(
            "LOUD-WARN [MaskAuthority]: discarded unverified mask provenance "
            "classification_id=%s model=%s",
            region.classification_id,
            region.classifier_model,
        )
    return valid
