from dataclasses import dataclass
import re


@dataclass(frozen=True)
class TranslationQualityResult:
    accepted: bool
    status: str
    reasons: tuple[str, ...]
    cjk_ratio: float


class TranslationQualityGate:
    def __init__(
        self,
        *,
        max_cjk_ratio: float = 0.30,
        placeholder_patterns: tuple[str, ...] | None = None,
    ) -> None:
        if not 0.0 <= max_cjk_ratio <= 1.0:
            raise ValueError("max_cjk_ratio must be between 0 and 1.")
        self.max_cjk_ratio = max_cjk_ratio
        self.placeholder_patterns = placeholder_patterns or (r"^DỊCH_\d+$",)

    def evaluate(
        self, *, source_text: str, target_text: str
    ) -> TranslationQualityResult:
        reasons = []
        source = source_text.strip()
        target = target_text.strip()
        if not source:
            reasons.append("empty_source_text")
        if not target:
            reasons.append("empty_target_text")
        if target and any(
            re.fullmatch(pattern, target, flags=re.IGNORECASE)
            for pattern in self.placeholder_patterns
        ):
            reasons.append("placeholder_target_text")
        if target and "[review]" in target.casefold():
            reasons.append("review_marker_target_text")
        cjk_ratio = self._cjk_ratio(target)
        if target and cjk_ratio > self.max_cjk_ratio:
            reasons.append("excessive_cjk_in_target")
        return TranslationQualityResult(
            accepted=not reasons,
            status="accepted" if not reasons else "review_required",
            reasons=tuple(reasons),
            cjk_ratio=cjk_ratio,
        )

    @staticmethod
    def _cjk_ratio(text: str) -> float:
        letters = [char for char in text if char.isalpha()]
        if not letters:
            return 0.0
        cjk = sum(
            "\u3400" <= char <= "\u9fff"
            or "\uf900" <= char <= "\ufaff"
            or "\u3040" <= char <= "\u30ff"
            or "\uac00" <= char <= "\ud7af"
            for char in letters
        )
        return cjk / len(letters)
