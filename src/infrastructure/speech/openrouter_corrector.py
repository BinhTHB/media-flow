"""Evidence-based contextual ASR correction through OpenRouter.

The corrector is intentionally selective to protect rate limits:
- Primary ASR text is trusted by default.
- A secondary ASR is called only when suspicious transcript features are found.
- Only suspicious segments are eligible for text replacement.
- Segment ids/timestamps are never changed.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import List, Sequence

from src.core.contracts import ASRCorrectionPort, ASRPort
from src.core.models.speech_segment import SpeechSegment
from src.features.translation.text_quality import (
    is_collapsed_speech,
    is_effectively_identical,
    speech_defect_tokens,
    substitute_speech_defects,
)
from src.infrastructure.translation.openrouter_translation import (
    DEFAULT_FREE_MODELS,
    _split_values,
    configured_openrouter_keys,
)

logger = logging.getLogger(__name__)

_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_LATIN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,}")
_PROPER_NAME = re.compile(r"\b[A-Z][a-z]{2,}\b")
_CLAUSE_BREAK = re.compile(r"(?<=[.!?…,;])\s+")
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def _drops_proper_name(primary: str, replacement: str) -> bool:
    folded = replacement.casefold()
    return any(token.casefold() not in folded for token in _PROPER_NAME.findall(primary))


def _speakable_alternative(primary: str, alternative: str, duration: float) -> str:
    """Return a second-read clause that fits this slot, or an empty string.

    A two-second Groq fragment often overlaps a Gemini paragraph. The paragraph
    is kept as evidence for a token swap. Only a clause short enough to speak
    inside the slot may replace the line.
    """
    budget = max(len(primary) + 16, int(max(0.2, duration) * 22))
    pieces = [alternative]
    if len(alternative) > budget:
        pieces = [part.strip(" ,;") for part in _CLAUSE_BREAK.split(alternative) if part.strip(" ,;")]
    primary_tokens = {token.casefold() for token in _WORD.findall(primary)}
    primary_defects = len(speech_defect_tokens(primary))
    best = ""
    best_score = -1
    for piece in pieces:
        if not piece or len(piece) > budget:
            continue
        piece_defects = len(speech_defect_tokens(piece))
        if piece_defects and piece_defects >= primary_defects:
            continue
        score = len(primary_tokens & {token.casefold() for token in _WORD.findall(piece)})
        # One shared function word ("ở") is not evidence this clause is the same line.
        # A clause from the next sentence ("Trong lọ dầu") must not replace the
        # line that still contains the miss ("bánh tisa").
        same_phrase = piece == alternative or _keeps_defect_anchor(primary, piece)
        if score > best_score and same_phrase and (score >= 2 or piece == alternative):
            best_score = score
            best = piece
    return best


def _keeps_defect_anchor(primary: str, piece: str) -> bool:
    """True when the clause still talks about the word next to the miss.

    ``bánh tisa ở Ý trong lọ dầu`` may overlap the later clause
    ``Trong lọ dầu ô liu ở Đức``. That clause does not contain ``bánh``, so it
    is a different sentence and must not replace this slot.
    """
    tokens = _WORD.findall(primary)
    defects = {token.casefold() for token in speech_defect_tokens(primary)}
    if not defects:
        return True
    piece_cf = piece.casefold()
    anchors: list[str] = []
    for index, token in enumerate(tokens):
        if token.casefold() not in defects:
            continue
        for neighbor_index in (index - 1, index + 1):
            if neighbor_index < 0 or neighbor_index >= len(tokens):
                continue
            neighbor = tokens[neighbor_index]
            if neighbor.casefold() in defects or len(neighbor) < 2:
                continue
            anchors.append(neighbor.casefold())
            break
    if not anchors:
        return True
    return all(anchor in piece_cf for anchor in anchors)


class OpenRouterASRCorrector(ASRCorrectionPort):
    endpoint = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        api_keys: Sequence[str] | None = None,
        models: Sequence[str] | None = None,
        secondary_provider: ASRPort | None = None,
        timeout_seconds: float = 90.0,
    ) -> None:
        resolved_keys = configured_openrouter_keys() if api_keys is None else api_keys
        self.api_keys = [k.strip() for k in resolved_keys if k and k.strip()]
        env_models = _split_values(os.getenv("OPENROUTER_ASR_CORRECTION_MODELS"))
        self.models = list(models or env_models or DEFAULT_FREE_MODELS)
        self.secondary_provider = secondary_provider
        self.timeout_seconds = timeout_seconds
        self._cursor = 0

    def correct_segments(
        self,
        segments: List[SpeechSegment],
        language: str = "auto",
        audio_path: str | None = None,
    ) -> List[SpeechSegment]:
        if not segments:
            return segments

        suspicious = self._suspicious_indices(segments, language)
        if not suspicious:
            return segments

        secondary_segments: List[SpeechSegment] = []
        if self.secondary_provider is not None and audio_path:
            try:
                secondary_segments = self.secondary_provider.transcribe(audio_path, language=language)
            except Exception as exc:
                logger.warning("Secondary ASR failed; retaining primary transcript: %s", exc)
                return segments

        if secondary_segments:
            before = len(segments)
            segments = self._splice_collapsed(segments, secondary_segments)
            if len(segments) != before:
                logger.info(
                    "ASR: split a collapsed window into %d timed lines",
                    len(segments) - before + 1,
                )
            suspicious = self._suspicious_indices(segments, language)
            if not suspicious:
                return segments

        alternatives = (
            self._align_alternatives(segments, secondary_segments, suspicious)
            if secondary_segments
            else {}
        )
        repaired = self._repair_defective_lines(segments, alternatives)
        if repaired:
            logger.info("ASR: replaced %d defective line(s) from the secondary transcript", repaired)
            suspicious = self._suspicious_indices(segments, language)
        pending = list(suspicious)

        # Do not ask an LLM to invent a correction without independent audio evidence.
        if not pending or not self.api_keys or not self.models:
            return segments
        if not any(alternatives.get(index, "").strip() for index in pending):
            return segments

        payload_base = self._build_payload(segments, pending, alternatives, language)
        errors: list[str] = []
        for model in self.models:
            for offset in range(len(self.api_keys)):
                key_index = (self._cursor + offset) % len(self.api_keys)
                try:
                    payload = dict(payload_base)
                    payload["model"] = model
                    text = self._post(payload, self.api_keys[key_index])
                    replacements = self._parse_response(text, pending)
                    # A response is not a successful correction if it preserves the
                    # exact suspicious failure mode while independent secondary ASR
                    # provides a native-script alternative. Reject it and continue
                    # to the next model/key instead of making correction stochastic.
                    for idx, corrected_text in replacements.items():
                        primary_text = segments[idx].source_text or ""
                        secondary_text = alternatives.get(idx, "") or ""
                        primary_mixed = bool(_CJK_RE.search(primary_text) and _LATIN_TOKEN_RE.search(primary_text))
                        secondary_native = bool(_CJK_RE.search(secondary_text) and not _LATIN_TOKEN_RE.search(secondary_text))
                        corrected_mixed = bool(_CJK_RE.search(corrected_text) and _LATIN_TOKEN_RE.search(corrected_text))
                        if primary_mixed and secondary_native and corrected_mixed:
                            raise RuntimeError(
                                "consensus preserved suspicious mixed CJK+Latin despite native-script secondary evidence"
                            )
                    self._cursor = (key_index + 1) % len(self.api_keys)
                    for idx, corrected_text in replacements.items():
                        if corrected_text:
                            segments[idx].source_text = corrected_text
                    return segments
                except Exception as exc:
                    errors.append(f"{model} key#{key_index + 1}: {str(exc)[:180]}")

        logger.warning("ASR consensus failed open; primary transcript retained. %s", " | ".join(errors[-4:]))
        return segments

    @staticmethod
    def _suspicious_indices(segments: List[SpeechSegment], language: str) -> List[int]:
        result: list[int] = []
        for idx, segment in enumerate(segments):
            text = segment.source_text or ""
            # Mixed CJK + Latin tokens are worth validating, but not automatically replacing.
            # This catches errors such as Chinese 快递/外卖 being decoded as "Wi-Fi".
            if _CJK_RE.search(text) and _LATIN_TOKEN_RE.search(text):
                result.append(idx)
                continue
            if is_collapsed_speech(text):
                result.append(idx)
                continue
            if speech_defect_tokens(text):
                result.append(idx)
                continue
            logprob = segment.asr_logprob
            if logprob is not None and logprob < -0.8:
                result.append(idx)
                continue
            if "�" in text or "<?>" in text:
                result.append(idx)
        return result

    @staticmethod
    def _align_alternatives(
        primary: List[SpeechSegment],
        secondary: List[SpeechSegment],
        suspicious: Sequence[int],
    ) -> dict[int, str]:
        alternatives: dict[int, str] = {}
        for idx in suspicious:
            item = primary[idx]
            overlaps: list[tuple[float, SpeechSegment]] = []
            for candidate in secondary:
                overlap = min(item.end_time, candidate.end_time) - max(item.start_time, candidate.start_time)
                if overlap > 0.05:
                    overlaps.append((overlap, candidate))
                    continue
                # A defect often sits one slot after the sentence that names the
                # real word ("bánh pizza" ends, then the next slot still says "tisa").
                if candidate.end_time <= item.start_time:
                    gap = item.start_time - candidate.end_time
                elif candidate.start_time >= item.end_time:
                    gap = candidate.start_time - item.end_time
                else:
                    continue
                if gap <= 1.5:
                    overlaps.append((0.0, candidate))
            overlaps.sort(key=lambda pair: (pair[1].start_time, -pair[0]))
            text = " ".join(seg.source_text.strip() for _, seg in overlaps if seg.source_text.strip()).strip()
            if text and text != item.source_text.strip():
                alternatives[idx] = text
        return alternatives

    @staticmethod
    def _splice_collapsed(
        primary: List[SpeechSegment],
        secondary: Sequence[SpeechSegment],
    ) -> List[SpeechSegment]:
        """Replace a vowel-stripped window with the timed lines that overlap it.

        One Groq window can cover several spoken sentences. Joining the
        secondary transcript into that window paints every sentence at once.
        Each overlapping secondary line keeps its own start and end.
        """
        assigned: dict[int, list[SpeechSegment]] = {index: [] for index in range(len(primary))}
        for piece in secondary:
            best_index = None
            best_overlap = 0.05
            for index, item in enumerate(primary):
                overlap = min(item.end_time, piece.end_time) - max(item.start_time, piece.start_time)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_index = index
            if best_index is not None:
                assigned[best_index].append(piece)

        result: list[SpeechSegment] = []
        for index, item in enumerate(primary):
            if not is_collapsed_speech(item.source_text or ""):
                result.append(item)
                continue
            readable = [
                piece
                for piece in assigned[index]
                if (piece.source_text or "").strip() and not is_collapsed_speech(piece.source_text)
            ]
            readable.sort(key=lambda piece: (piece.start_time, piece.end_time))
            if not readable:
                result.append(item)
                continue
            if len(readable) == 1:
                item.source_text = readable[0].source_text.strip()
                result.append(item)
                continue
            pieces: list[SpeechSegment] = []
            for number, piece in enumerate(readable):
                start = max(item.start_time, float(piece.start_time))
                end = min(item.end_time, float(piece.end_time))
                if end - start < 0.2:
                    continue
                pieces.append(
                    SpeechSegment(
                        segment_id=f"{item.segment_id}_{number}",
                        start_time=start,
                        end_time=end,
                        source_text=piece.source_text.strip(),
                        source_language=piece.source_language or item.source_language,
                        speaker_id=piece.speaker_id or item.speaker_id,
                    )
                )
            if len(pieces) >= 2:
                result.extend(pieces)
            else:
                item.source_text = " ".join(piece.source_text.strip() for piece in readable)
                result.append(item)
        return result

    @staticmethod
    def _repair_defective_lines(
        segments: List[SpeechSegment],
        alternatives: dict[int, str],
    ) -> int:
        """Replace a readable-but-wrong line when the second transcript lacks that error.

        A second read that overlaps several sentences is evidence, not a caption.
        A short slot keeps its own wording and only swaps the broken token.
        A failed second read leaves the primary line in place. Only a collapsed
        line, which is not speakable, is dropped later.
        """
        repaired = 0
        for index, item in enumerate(segments):
            primary = item.source_text or ""
            if is_collapsed_speech(primary):
                continue
            alternative = (alternatives.get(index) or "").strip()
            if not alternative or is_collapsed_speech(alternative):
                continue
            duration = max(0.2, item.end_time - item.start_time)
            candidate = _speakable_alternative(primary, alternative, duration)
            if candidate and _drops_proper_name(primary, candidate):
                candidate = ""
            defects = speech_defect_tokens(primary)
            low_confidence = item.asr_logprob is not None and item.asr_logprob < -0.8
            candidate_defects = speech_defect_tokens(candidate) if candidate else defects
            better = bool(candidate) and bool(defects) and len(candidate_defects) < len(defects)
            unsure = (
                bool(candidate)
                and low_confidence
                and not candidate_defects
                and not is_effectively_identical(primary, candidate)
            )
            if better or unsure:
                item.source_text = candidate
                repaired += 1
                continue
            substituted = substitute_speech_defects(primary, alternative)
            if substituted != primary and not _drops_proper_name(primary, substituted):
                item.source_text = substituted
                repaired += 1
        return repaired

    @staticmethod
    def _build_payload(
        segments: List[SpeechSegment],
        suspicious: Sequence[int],
        alternatives: dict[int, str],
        language: str,
    ) -> dict:
        scene = [
            {"id": i, "start": round(seg.start_time, 3), "end": round(seg.end_time, 3), "primary": seg.source_text}
            for i, seg in enumerate(segments)
        ]
        candidates = [
            {
                "id": idx,
                "primary": segments[idx].source_text,
                "secondary": alternatives.get(idx, ""),
            }
            for idx in suspicious
            if alternatives.get(idx, "")
        ]
        prompt = (
            "Resolve ASR disagreements for one continuous video scene.\n"
            f"Language hint: {language}.\n"
            "Primary timestamps/segmentation are authoritative. Only choose the most likely spoken TEXT.\n"
            "Use phonetic plausibility, grammar, and surrounding scene context.\n"
            "Rules:\n"
            "- Do NOT translate. Keep the source language/script.\n"
            "- Only ids listed in candidates may change.\n"
            "- Prefer an existing primary or secondary candidate; make a tiny correction only if both clearly contain the same obvious recognition error.\n"
            "- Mixed CJK + unexpected Latin tokens are a known ASR failure mode. If primary contains such a token but secondary provides a natural native-script phrase that fits the surrounding verb/context, prefer secondary unless the Latin token is clearly a supported proper name or technical term.\n"
            "- Preserve names, numbers, negation, repetitions, slang and short utterances.\n"
            "- If evidence is genuinely ambiguous after contextual/grammatical comparison, keep primary exactly.\n"
            "Return strict JSON only: {\"corrections\":[{\"id\":5,\"text\":\"...\"}]}.\n\n"
            f"Full scene: {json.dumps(scene, ensure_ascii=False)}\n"
            f"Disagreements: {json.dumps(candidates, ensure_ascii=False)}"
        )
        return {
            "temperature": 0,
            "messages": [
                {"role": "system", "content": "Evidence-based ASR consensus. JSON only."},
                {"role": "user", "content": prompt},
            ],
        }

    def _post(self, payload: dict, api_key: str) -> str:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "MediaFlow/1.0",
        }
        app_title = os.getenv("OPENROUTER_APP_TITLE", "MediaFlow").strip()
        referer = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
        if app_title:
            headers["X-Title"] = app_title
        if referer:
            headers["HTTP-Referer"] = referer
        request = urllib.request.Request(self.endpoint, data=body, method="POST", headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload_out = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail[:300]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"network error: {exc.reason}") from exc

        choices = payload_out.get("choices") if isinstance(payload_out, dict) else None
        if not choices:
            raise RuntimeError("OpenRouter ASR consensus returned no choices")
        content = choices[0].get("message", {}).get("content", "")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("OpenRouter ASR consensus returned empty content")
        return content.strip()

    @staticmethod
    def _parse_response(content: str, suspicious: Sequence[int]) -> dict[int, str]:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lstrip().startswith("json"):
                cleaned = cleaned.lstrip()[4:].lstrip()
        data = json.loads(cleaned)
        items = data.get("corrections") if isinstance(data, dict) else None
        if not isinstance(items, list):
            raise RuntimeError("ASR consensus response missing corrections[]")

        allowed = set(suspicious)
        result: dict[int, str] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            idx = item.get("id")
            text = item.get("text")
            if isinstance(idx, int) and idx in allowed and isinstance(text, str) and text.strip():
                result[idx] = text.strip()
        if not result:
            raise RuntimeError("ASR consensus returned no usable correction")
        return result
