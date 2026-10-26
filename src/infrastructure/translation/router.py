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

                suspicious = self._suspicious_indices(texts, result, target_language)
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
                            if idx not in self._suspicious_indices(texts, candidate, target_language):
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

                # One unresolvable line must not destroy its chunk neighbors:
                # return best-effort results and leave only the failed ids
                # empty. Callers map empty translations to per-event fail-safe
                # statuses (TRANSLATION_EMPTY), so the damage stays local.
                if unresolved:
                    logger.warning(
                        "Translation semantic validator unresolved ids %s after isolated repairs; failing only those lines",
                        unresolved,
                    )
                    for idx in unresolved:
                        repaired[idx] = ""
                    return repaired
                return repaired
            except Exception as exc:
                logger.warning("Translation provider %s failed; trying fallback: %s", name, exc)
                errors.append(f"{name}: {str(exc)[:240]}")

        raise RuntimeError("All translation providers failed. " + " | ".join(errors))

    @staticmethod
    def _validate_complete(result: List[str], texts: List[str]) -> None:
        if len(result) != len(texts) or any(not str(item).strip() for item in result):
            raise RuntimeError("provider returned incomplete translation batch")

    # Vietnamese precomposed diacritics (Latin Extended Additional + Đ đ + the
    # tone-marked Latin-1 vowels). A Vietnamese caption translation almost
    # always carries at least one; "describe mode" output ("Description:
    # \"Soft eyes\"") and English glosses carry none.
    _VN_DIACRITIC_RE = re.compile(
        r"[\u00C0-\u00FF\u0102\u0103\u0110\u0111\u01A0\u01A1\u01AF\u01B0\u1EA0-\u1EF9]"
    )

    # English words that mark a sentence or gloss label rather than a proper
    # noun. A name-shaped answer ("Bartholomew Kuma", "One Piece") never
    # contains them; describe-mode answers ("Made with AI") do.
    _EN_SENTENCE_WORDS = frozenset(
        {
            "the", "a", "an", "of", "and", "or", "is", "are", "was", "were",
            "with", "by", "made", "produced", "created", "generated",
            "description", "meaning", "translation", "note", "explanation",
            "gloss", "subtitle", "dubbed", "subtitled",
        }
    )

    @classmethod
    def _is_name_shaped(cls, text: str) -> bool:
        """True when the text looks like a proper noun / term instead of an
        English sentence: 1-4 words, each capitalized, an all-caps acronym
        (<=4 chars), or a number."""
        words = re.findall(r"[A-Za-z0-9']+", text)
        if not 1 <= len(words) <= 4:
            return False
        for word in words:
            if word.lower() in cls._EN_SENTENCE_WORDS:
                return False
            if word.isdigit():
                continue
            if word.isupper() and len(word) <= 4:
                continue
            if word[0].isupper():
                continue
            return False
        return True

    @classmethod
    def _suspicious_indices(
        cls, source: List[str], translated: List[str], target_language: str = "vi"
    ) -> List[int]:
        suspicious: list[int] = []
        vi_target = str(target_language or "").lower().startswith("vi")

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

            # A model echoing task instructions instead of translating yields
            # prompt boilerplate no real caption contains ("Preserve line
            # numbers (`i: translated_text`)"). CJK echoes are invisible to
            # _normalize, so this is also the metadata-detection net for them.
            if re.search(
                r"translated_text|preserve (?:the )?line numbers?|numbered (?:lines?|segments?)",
                str(current),
                re.IGNORECASE,
            ):
                suspicious.append(idx)
                continue

            # "Describe mode" prefixes: the model annotates the source instead
            # of translating it ("Description: \"Soft eyes\"").
            if re.search(
                r"^(?:description|instruction|meaning|explanation|note|context|gloss)\s*:",
                str(current).strip(),
                re.IGNORECASE,
            ):
                suspicious.append(idx)
                continue

            # Untranslated source echo: a response in "explain" mode returns
            # "<source> (Chinese: <english gloss>)". _normalize strips CJK, so
            # the copied source is invisible to every check below unless the
            # CJK characters themselves are compared. Vietnamese output never
            # contains CJK, so any shared CJK run with the source means the
            # line was not translated (conservative for Latin-script targets).
            current_cjk = set(re.findall(r"[\u3400-\u9fff]", str(current)))
            if current_cjk:
                source_cjk = set(re.findall(r"[\u3400-\u9fff]", str(source[idx])))
                if len(current_cjk & source_cjk) >= 2:
                    suspicious.append(idx)
                    continue

            # Language-meta annotations ("(Chinese: ...)", "(English)") mark
            # transcription glosses, not caption translations.
            if re.search(
                r"\((?:chinese|english|pinyin|mandarin|cantonese|hanzi|hanyu)\s*[):]",
                str(current),
                re.IGNORECASE,
            ):
                suspicious.append(idx)
                continue

            # Target-language sanity: a Vietnamese translation carries
            # Vietnamese diacritics. Model "describe mode" answers
            # ("Description: \"Soft eyes\""), English glosses, and untranslated
            # sentences have none and must not render as READY captions.
            # Allowed exceptions: proper-noun passthrough of a short source
            # line, and short lines whose content tokens survive translation
            # (brand transliterations like "@抖音: Spongecake" -> "@Douyin:
            # Spongecake").
            if (
                vi_target
                and not cls._VN_DIACRITIC_RE.search(str(current))
                and re.search(r"[A-Za-z]", str(current))
            ):
                src_str = str(source[idx])
                identical = current_norm == cls._normalize(src_str)
                source_tokens = set(re.findall(r"[a-z0-9]{3,}", src_str.lower()))
                current_tokens = set(re.findall(r"[a-z0-9]{3,}", str(current).lower()))
                short_shared = (
                    len(str(current).split()) <= 4
                    and len(src_str.split()) <= 6
                    and bool(source_tokens & current_tokens)
                )
                # Proper-noun passthrough: CJK person/character names are
                # canonically rendered in Latin without Vietnamese diacritics
                # ("巴索罗米·熊" -> "Bartholomew Kuma"). The CJK middle dot is
                # a strong person-name marker, and a capitalized term-shaped
                # answer with no English sentence words is a name, not a
                # describe-mode gloss ("Made with AI" still fails below).
                name_passthrough = (
                    "\u00b7" in src_str
                    or "\u30fb" in src_str
                    or cls._is_name_shaped(str(current))
                )
                if (
                    not (identical and len(src_str.split()) <= 4)
                    and not short_shared
                    and not name_passthrough
                ):
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
            # Drop-protection applies only to brand-like tokens (capitalized
            # >=5 chars, all-caps, or containing a digit): bilingual CJK
            # captions carry an English gloss ("第一条: 要温柔\nOne: be
            # gentle") whose ordinary words legitimately disappear once the CJK
            # meaning is translated into Vietnamese.
            current_lower = str(current).lower()
            drop_protected = {
                token.lower()
                for token in re.findall(r"[A-Za-z][A-Za-z0-9._-]{2,}", str(source[idx]))
                if re.match(r"^[A-Z][A-Za-z0-9]{4,}$", token)
                or re.match(r"^[A-Z0-9]{3,}$", token)
                or re.search(r"\d", token)
            }
            if any(token not in current_lower for token in drop_protected):
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
