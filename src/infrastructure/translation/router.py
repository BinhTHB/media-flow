"""Failover + lightweight semantic validation router for TranslationPort implementations."""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import List, Sequence, Tuple

from src.core.contracts import TranslationPort

logger = logging.getLogger(__name__)


class FallbackTranslationAdapter(TranslationPort):
    def __init__(self, providers: Sequence[Tuple[str, TranslationPort]]) -> None:
        self.providers = [(name, provider) for name, provider in providers if provider is not None]

    def translate_texts(
        self,
        texts: List[str],
        target_language: str = "vi",
        source_language: str = "auto",
    ) -> List[str]:
        if not texts:
            return []
        if not self.providers:
            raise RuntimeError("No translation provider is configured.")

        errors: List[str] = []
        for provider_index, (name, provider) in enumerate(self.providers):
            try:
                result = provider.translate_texts(
                    texts=texts,
                    target_language=target_language,
                    source_language=source_language,
                )
                self._validate_complete(result, texts)

                suspicious = self._suspicious_indices(texts, result)
                if not suspicious:
                    return result

                # Quality escalation is selective and context-stripping: retranslate
                # only suspicious ids in isolation so neighboring lines cannot leak.
                # Try the same cheap/free provider first, then later fallbacks.
                repaired = list(result)
                unresolved = list(suspicious)
                repair_providers = [(name, provider), *self.providers[provider_index + 1 :]]
                for repair_name, repair_provider in repair_providers:
                    next_unresolved: list[int] = []
                    for idx in unresolved:
                        try:
                            single = repair_provider.translate_texts(
                                texts=[texts[idx]],
                                target_language=target_language,
                                source_language=source_language,
                            )
                            self._validate_complete(single, [texts[idx]])
                            candidate = list(repaired)
                            candidate[idx] = single[0]
                            if idx not in self._suspicious_indices(texts, candidate):
                                repaired[idx] = single[0]
                            else:
                                next_unresolved.append(idx)
                        except Exception as exc:
                            errors.append(f"{repair_name} isolated-repair id={idx}: {str(exc)[:240]}")
                            next_unresolved.append(idx)
                    unresolved = next_unresolved
                    if not unresolved:
                        logger.warning(
                            "Translation semantic validator isolated-repaired ids %s using %s",
                            suspicious,
                            repair_name,
                        )
                        return repaired

                raise RuntimeError(f"semantic validator rejected ids {unresolved}")
            except Exception as exc:
                logger.warning("Translation provider %s failed; trying fallback: %s", name, exc)
                errors.append(f"{name}: {str(exc)[:240]}")

        raise RuntimeError("All translation providers failed. " + " | ".join(errors))

    @staticmethod
    def _validate_complete(result: List[str], texts: List[str]) -> None:
        if len(result) != len(texts) or any(not str(item).strip() for item in result):
            raise RuntimeError("provider returned incomplete translation batch")

    @classmethod
    def _suspicious_indices(cls, source: List[str], translated: List[str]) -> List[int]:
        suspicious: list[int] = []

        # Cross-id anchors that should never migrate between neighboring subtitle
        # rows. Restrict proper-token protection to mixed CJK+Latin source lines
        # (Bloomberg终端, WiFi..., etc.) so ordinary all-English captions can still
        # be naturally translated rather than copied word-for-word.
        number_owners: dict[str, set[int]] = {}
        protected_token_owners: dict[str, set[int]] = {}
        source_numbers: list[set[str]] = []
        source_protected_tokens: list[set[str]] = []
        for source_idx, raw_source in enumerate(source):
            nums = set(re.findall(r"(?<!\d)\d+(?:[.,]\d+)?", str(raw_source)))
            source_numbers.append(nums)
            for number in nums:
                number_owners.setdefault(number, set()).add(source_idx)

            has_cjk = bool(re.search(r"[\u3400-\u9fff]", str(raw_source)))
            latin_tokens: set[str] = set()
            if has_cjk:
                latin_tokens = {
                    token.lower()
                    for token in re.findall(r"[A-Za-z][A-Za-z0-9._-]{2,}", str(raw_source))
                }
            source_protected_tokens.append(latin_tokens)
            for token in latin_tokens:
                protected_token_owners.setdefault(token, set()).add(source_idx)

        for idx, current in enumerate(translated):
            current_norm = cls._normalize(current)
            if not current_norm:
                suspicious.append(idx)
                continue

            # Arabic numbers already present in the source must stay in the same
            # segment. Also reject a number copied from a neighboring source id.
            output_numbers = set(re.findall(r"(?<!\d)\d+(?:[.,]\d+)?", str(current)))
            if not source_numbers[idx].issubset(output_numbers):
                suspicious.append(idx)
                continue
            leaked_number = any(
                number not in source_numbers[idx]
                and number in number_owners
                and idx not in number_owners[number]
                for number in output_numbers
            )
            if leaked_number:
                suspicious.append(idx)
                continue

            # Preserve mixed-script proper tokens in-place and reject migration to
            # another id (e.g. Bloomberg being attached to the previous sentence).
            current_lower = str(current).lower()
            if any(token not in current_lower for token in source_protected_tokens[idx]):
                suspicious.append(idx)
                continue
            leaked_token = any(
                token in current_lower
                and token not in source_protected_tokens[idx]
                # A token that also occurs in this line's own source is a shared
                # entity (e.g. an account name repeated across captions), not a
                # neighbor leak.
                and token not in str(source[idx]).lower()
                and idx not in owners
                for token, owners in protected_token_owners.items()
            )
            if leaked_token:
                suspicious.append(idx)
                continue

            # Small high-confidence domain guards for recurring Chinese finance
            # ambiguities seen in real footage. They do not prescribe the exact
            # Vietnamese wording; they only reject known wrong literal senses.
            raw_source = str(source[idx])
            if "国家队" in raw_source and (
                "doi quoc gia" in current_norm or "doi tuyen quoc gia" in current_norm
            ):
                suspicious.append(idx)
                continue
            if "燥热" in raw_source and "nong long" in current_norm:
                suspicious.append(idx)
                continue

            # Detect an LLM accidentally copying the previous segment into this id.
            if idx > 0 and source[idx].strip() != source[idx - 1].strip():
                previous_norm = cls._normalize(translated[idx - 1])
                if previous_norm and len(previous_norm) >= 4:
                    if current_norm.startswith(previous_norm + " "):
                        suspicious.append(idx)
                        continue

            # Detect repeated leading phrase such as "xin loi xin loi ..." when
            # the source itself does not contain adjacent repetition.
            if not cls._source_has_repetition(source[idx]):
                words = current_norm.split()
                for width in (1, 2, 3):
                    if len(words) >= width * 2 and words[:width] == words[width : width * 2]:
                        suspicious.append(idx)
                        break
        return sorted(set(suspicious))

    @staticmethod
    def _normalize(text: str) -> str:
        value = str(text).lower().replace("đ", "d")
        value = unicodedata.normalize("NFD", value)
        value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
        value = re.sub(r"[^a-z0-9\s]", " ", value)
        return re.sub(r"\s+", " ", value).strip()

    @staticmethod
    def _source_has_repetition(text: str) -> bool:
        compact = re.sub(r"[\s\W_]+", "", str(text), flags=re.UNICODE)
        if len(compact) < 2:
            return False
        max_width = min(6, len(compact) // 2)
        for width in range(1, max_width + 1):
            for start in range(0, len(compact) - 2 * width + 1):
                if compact[start : start + width] == compact[start + width : start + 2 * width]:
                    return True
        return False
