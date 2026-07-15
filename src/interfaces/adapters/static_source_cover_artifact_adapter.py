import json
from pathlib import Path
from typing import Mapping, Optional, Tuple

from src.application.services.dynamic_subtitle_timeline_service import (
    DynamicSourceInterval,
    TranslatedSourceSegment,
)


class StaticSourceCoverArtifactAdapter:
    """Adapter that parses the v2 static-source-cover baseline artifacts.

    Reads from a sample folder under output/static_source_cover_mvp_v2/<sample>/
    and emits typed Application objects (DynamicSourceInterval).
    No file mutation and no business decisions are made here.
    """

    def __init__(self, sample_dir: str):
        self.sample_dir = Path(sample_dir)
        self.cover_report = self._read_json("_static_cover/debug/cover_source_subtitle_report.json")
        self.interval_report = self._read_json("debug/interval_decision_report.json")
        self.provenance_report = self._read_json("debug/artifact_provenance_report.json")

    def _read_json(self, relative_path: str) -> dict:
        path = self.sample_dir / relative_path
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def load_intervals(self) -> list:
        captions = self.cover_report.get("captions", []) or []
        source_text_by_interval = self._index_source_text()
        results = []
        for cap in captions:
            interval_id = cap.get("interval_id") or cap.get("caption_id")
            time_range = cap.get("time_range") or {}
            source_bbox = cap.get("source_subtitle_bbox")
            cover_bbox = cap.get("mask_bbox")
            keep_reject = cap.get("keep_reject_reason")
            metadata = {
                "candidate_level_override": bool(
                    cap.get("static_region_person_overlap_override", False)
                ),
                "override_reason": cap.get("override_reason"),
                "static_region_person_overlap_override": bool(
                    cap.get("static_region_person_overlap_override", False)
                ),
                "local_safety_override": bool(cap.get("local_safety_override", False)),
                "person_overlap_score": cap.get("person_overlap_score"),
                "face_overlap_score": cap.get("face_overlap_score"),
                "watermark_overlap_score": cap.get("watermark_overlap_score"),
                "scene_text_overlap_score": cap.get("scene_text_overlap_score"),
            }
            results.append(
                DynamicSourceInterval(
                    interval_id=interval_id,
                    start=float(time_range.get("start", 0.0)),
                    end=float(time_range.get("end", 0.0)),
                    source_text=source_text_by_interval.get(interval_id, ""),
                    source_bbox=source_bbox,
                    expanded_cover_bbox=cover_bbox,
                    safety_status="skipped" if keep_reject else "safe",
                    safety_reason=keep_reject,
                    metadata=metadata,
                )
            )
        return results

    def _index_source_text(self) -> Mapping[str, str]:
        mapping = {}
        for item in self.interval_report.get("intervals", []) or []:
            cand = item.get("selected_candidate") or {}
            text = cand.get("normalized_text") or cand.get("raw_ocr_text") or ""
            if text:
                mapping[item.get("interval_id")] = text
        return mapping

    def load_segments(
        self, target_text_map: Optional[Mapping[str, str]] = None,
        provenance: str = "ocr_source_text",
    ) -> list:
        target_text_map = target_text_map or {}
        results = []
        for cap in self.cover_report.get("captions", []) or []:
            interval_id = cap.get("interval_id") or cap.get("caption_id")
            time_range = cap.get("time_range") or {}
            results.append(
                TranslatedSourceSegment(
                    segment_id=interval_id,
                    start=float(time_range.get("start", 0.0)),
                    end=float(time_range.get("end", 0.0)),
                    source_text=cap.get("source_subtitle_text", "") or "",
                    target_text=target_text_map.get(interval_id, ""),
                    provenance=provenance,
                    metadata={},
                )
            )
        return results

    def get_frame_dimensions(self) -> Tuple[int, int]:
        resolution = self.provenance_report.get("resolution") or {}
        return int(resolution.get("width", 0)), int(resolution.get("height", 0))