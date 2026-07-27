from src.application.services.ass_generator import ASSGenerator, _has_mask_provenance
from src.application.services.mask_authority import DecisionStore, MaskAuthority
from src.domain.entities.segment import Segment
from src.domain.entities.subtitle_region import SubtitleRegion
from src.domain.entities.subtitle_style import SubtitleStyle


def _region(
    classification_id: str | None = None,
    model: str | None = None,
) -> SubtitleRegion:
    return SubtitleRegion(
        0.0,
        1.0,
        0.1,
        0.7,
        0.4,
        0.1,
        0.9,
        classification_id=classification_id,
        is_subtitle=True,
        classifier_model=model,
    )


def _segment() -> Segment:
    return Segment(
        id="s",
        start=0.0,
        end=1.0,
        text="source",
        translated_text="dịch",
        source="ocr",
    )


def test_ocr_fallback_model_is_discarded() -> None:
    store = DecisionStore("run-1")
    region = _region("ocr-segment-1", "ocr_fallback")

    assert not _has_mask_provenance(region, store)
    ass = ASSGenerator.generate(
        [_segment()], SubtitleStyle(), mask_regions=[region], decision_store=store
    )
    assert "m 0 0" not in ass


def test_missing_decision_store_record_is_discarded() -> None:
    store = DecisionStore("run-1")
    region = _region("missing-decision", "v21_pipeline")

    assert not _has_mask_provenance(region, store)


def test_negative_decision_store_verdict_is_discarded() -> None:
    store = DecisionStore("run-1")
    classification_id = "negative-decision"
    store.record(
        classification_id=classification_id,
        model="v21_pipeline",
        verdict=False,
        raw_response_id="response-negative",
        cache_hit=False,
        live_or_cache="live",
    )
    region = _region(classification_id, "v21_pipeline")

    assert not _has_mask_provenance(region, store)


def test_verified_positive_decision_renders_mask() -> None:
    store = DecisionStore("run-1")
    authority = MaskAuthority(store)
    region = authority.decide(
        {
            "start_time": 0.0,
            "end_time": 1.0,
            "x": 0.1,
            "y": 0.7,
            "width": 0.4,
            "height": 0.1,
            "confidence": 0.9,
            "track_id": "T001",
            "proposal_ids": ["F00000_C01"],
        },
        {"is_subtitle": True, "category": "subtitle", "confidence": 0.95},
        model="v21_pipeline",
        raw_response_id="response-1",
    )

    assert region is not None
    assert _has_mask_provenance(region, store)
    ass = ASSGenerator.generate(
        [_segment()], SubtitleStyle(), mask_regions=[region], decision_store=store
    )
    assert "m 0 0" in ass


def test_negative_decision_never_creates_mask_region() -> None:
    store = DecisionStore("run-1")
    authority = MaskAuthority(store)

    region = authority.decide(
        {
            "start_time": 0.0,
            "end_time": 1.0,
            "x": 0.1,
            "y": 0.7,
            "width": 0.4,
            "height": 0.1,
            "confidence": 0.9,
        },
        {"is_subtitle": False},
        model="v21_pipeline",
    )

    assert region is None
    assert store.positive_count() == 0
