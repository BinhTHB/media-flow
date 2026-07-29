from src.application.services.ass_generator import ASSGenerator, _has_mask_provenance
from src.application.services.mask_authority import DecisionStore
from src.domain.entities.segment import Segment
from src.domain.entities.subtitle_region import SubtitleRegion
from src.domain.entities.subtitle_style import SubtitleStyle


def _region(track_id: str | None) -> SubtitleRegion:
    return SubtitleRegion(0.0, 1.0, 0.1, 0.7, 0.4, 0.1, 0.9, track_id=track_id)


def _segment() -> Segment:
    return Segment(id="s", start=0.0, end=1.0, text="source", translated_text="d?ch", source="ocr")


def _store(tmp_path, run_id: str = "run-1") -> DecisionStore:
    return DecisionStore(run_id, path=tmp_path / "mask_decisions.jsonl")


def test_missing_track_decision_is_discarded(tmp_path) -> None:
    store = _store(tmp_path)

    assert not _has_mask_provenance(_region("T001"), store)


def test_different_run_is_discarded(tmp_path) -> None:
    store = _store(tmp_path, "run-1")
    response = tmp_path / "responses" / "request_01.json"
    response.parent.mkdir()
    response.write_text("[]", encoding="utf-8")
    store.record(track_id="T001", is_subtitle=True, model="gemini-3.5-flash-lite", response_ref="responses/request_01.json", cache_hit=False)

    assert not _has_mask_provenance(_region("T001"), store, expected_run_id="run-2")


def test_negative_track_decision_is_discarded(tmp_path) -> None:
    store = _store(tmp_path)
    response = tmp_path / "responses" / "request_01.json"
    response.parent.mkdir()
    response.write_text("[]", encoding="utf-8")
    store.record(track_id="T001", is_subtitle=False, model="gemini-3.5-flash-lite", response_ref="responses/request_01.json", cache_hit=False)

    assert not _has_mask_provenance(_region("T001"), store)


def test_missing_response_ref_is_discarded(tmp_path) -> None:
    store = _store(tmp_path)
    store.record(track_id="T001", is_subtitle=True, model="gemini-3.5-flash-lite", response_ref="responses/missing.json", cache_hit=False)

    assert not _has_mask_provenance(_region("T001"), store)


def test_positive_track_decision_renders_all_its_masks(tmp_path) -> None:
    store = _store(tmp_path)
    response = tmp_path / "responses" / "request_01.json"
    response.parent.mkdir()
    response.write_text("[]", encoding="utf-8")
    store.record(track_id="T001", is_subtitle=True, model="gemini-3.5-flash-lite", response_ref="responses/request_01.json", cache_hit=False)
    regions = [_region("T001"), SubtitleRegion(1.0, 2.0, 0.2, 0.7, 0.4, 0.1, 0.9, track_id="T001")]

    assert all(_has_mask_provenance(region, store) for region in regions)
    ass = ASSGenerator.generate([_segment()], SubtitleStyle(), mask_regions=regions, decision_store=store)
    assert ass.count("m 0 0") == 2
