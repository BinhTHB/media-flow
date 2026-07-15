import pytest

from src.application.services.translation_quality_gate import TranslationQualityGate


@pytest.mark.parametrize(
    ("source", "target", "accepted", "reasons"),
    [
        ("你好", "Xin chào Việt Nam.", True, ()),
        ("你好", "   ", False, ("empty_target_text",)),
        ("你好", "DỊCH_1", False, ("placeholder_target_text",)),
        ("你好", "DỊCH_12", False, ("placeholder_target_text",)),
        ("你好", "[REVIEW]", False, ("review_marker_target_text",)),
        ("你好", "Cần [review] lại", False, ("review_marker_target_text",)),
        (" ", "Xin chào", False, ("empty_source_text",)),
    ],
)
def test_quality_gate_returns_stable_reasons(
    source: str, target: str, accepted: bool, reasons: tuple[str, ...]
) -> None:
    result = TranslationQualityGate().evaluate(
        source_text=source, target_text=target
    )

    assert result.accepted is accepted
    assert result.reasons == reasons
    assert result.status == ("accepted" if accepted else "review_required")


def test_quality_gate_allows_small_cjk_proper_name_ratio() -> None:
    result = TranslationQualityGate(max_cjk_ratio=0.30).evaluate(
        source_text="欢迎李明来到河内",
        target_text="Chào mừng 李 Minh đến Hà Nội",
    )

    assert result.accepted is True
    assert 0 < result.cjk_ratio < 0.30


def test_quality_gate_rejects_excessive_cjk_ratio() -> None:
    result = TranslationQualityGate(max_cjk_ratio=0.30).evaluate(
        source_text="你好世界",
        target_text="你好世界 xin chào",
    )

    assert result.accepted is False
    assert result.reasons == ("excessive_cjk_in_target",)


def test_quality_gate_handles_punctuation_only_without_division_by_zero() -> None:
    result = TranslationQualityGate().evaluate(
        source_text="你好", target_text="...!?"
    )

    assert result.accepted is True
    assert result.cjk_ratio == 0.0


def test_quality_gate_validates_config() -> None:
    with pytest.raises(ValueError, match="max_cjk_ratio"):
        TranslationQualityGate(max_cjk_ratio=1.1)
