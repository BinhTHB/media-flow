"""
Gemini Translation Infrastructure Adapter (Clean Architecture V2)

Implements TranslationPort using Gemini API directly via GeminiRuntime.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Optional

from src.core.contracts import TranslationPort
from src.infrastructure.gemini.contracts import GeminiRequest
from src.infrastructure.gemini.exceptions import GeminiAuthError, GeminiError
from src.infrastructure.gemini.runtime import GeminiRuntime

logger = logging.getLogger(__name__)


class GeminiTranslationAdapter(TranslationPort):
    def __init__(
        self,
        runtime: Optional[GeminiRuntime] = None,
        model: Optional[str] = None,
        purpose: str = "subtitle",
    ):
        self._runtime = runtime or GeminiRuntime.create()
        # MEDIAFLOW_GEMINI_MODEL overrides every Gemini call site (see
        # GeminiOCRAdapter for the matching OCR-side override).
        self.model = os.getenv("MEDIAFLOW_GEMINI_MODEL") or model or "gemini-3.5-flash-lite"
        self.purpose = purpose if purpose in {"subtitle", "spoken"} else "subtitle"

    def translate_texts(
        self, texts: List[str], target_language: str = "vi", source_language: str = "auto"
    ) -> List[str]:
        if not texts:
            return []

        if hasattr(self, "provider") and self.provider is not None:
            # Delegate to provider mock if injected in tests
            class DummySeg:
                def __init__(self, t):
                    self.text = t
                    self.translated_text = t
                    self.bbox = None

            segs = [DummySeg(t) for t in texts]
            res = self.provider.translate(segs, target_language=target_language)
            return [getattr(s, "translated_text", getattr(s, "text", "")) for s in res]

        chunk_size = 100
        translated_texts: List[str] = []
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i : i + chunk_size]
            chunk_translations = self._translate_chunk(chunk, target_language)
            translated_texts.extend(chunk_translations)

        return translated_texts

    def _translate_chunk(self, texts: List[str], target_language: str) -> List[str]:
        numbered_lines = "\n".join(f"{index}: {text}" for index, text in enumerate(texts))
        if self.purpose == "spoken":
            guidelines = (
                "1. These are consecutive speech segments from one video scene; use surrounding lines as context.\n"
                "2. Preserve every meaning-bearing detail, including short utterances, greetings, names, numbers, negation, objects, and meaningful repetition.\n"
                "3. Make the result natural when spoken aloud and concise enough for dubbing, but never omit content merely to shorten it.\n"
                "4. Do not merge or split numbered segments.\n"
            )
            prompt = (
                f"Translate each numbered video line into natural, conversational {target_language}.\n"
                "Translation guidelines:\n"
                f"{guidelines}"
                "7. Preserve every line number using the exact format 'i: translated_text'.\n"
                "8. Return only translated numbered lines with no explanations or extra quotes.\n\n"
                f"{numbered_lines}"
            )
        else:
            # Compact prompt with a worked example. Long rule-list prompts push
            # gemma-4-26b into "describe mode" (it annotates each line instead
            # of translating); a short instruction plus one example keeps it in
            # translation mode. Domain-quality guards live in the router
            # validator, not in the prompt.
            prompt = (
                f"Translate each numbered video subtitle line into natural, conversational {target_language}.\n"
                "Keep each line's meaning only, keep numbers, keep proper names.\n"
                "A line may contain the same caption twice (Chinese row plus its English translation): translate that meaning ONCE, as a single line.\n"
                "Output ONLY numbered lines in the format 'i: translation'.\n\n"
                "Example:\n0: 第一条: 要温柔\n0: Thứ nhất: Hãy dịu dàng\n\n"
                f"{numbered_lines}"
            )

        try:
            client = self._runtime.get_client(model=self.model)
            response = client.generate_content(
                GeminiRequest(
                    contents=[{"parts": [{"text": prompt}]}],
                    generation_config={"temperature": 0.1},
                    source_kind="translation",
                )
            )
            return self._parse_numbered_lines(response.text.strip(), texts)
        except GeminiAuthError as exc:
            logger.error("GEMINI_API_KEY environment variable is not set.")
            raise RuntimeError("GEMINI_API_KEY environment variable is not set.") from exc
        except GeminiError as exc:
            logger.error(f"Gemini translation failed: {exc}")
            raise RuntimeError(f"Gemini translation failed: {exc}") from exc

    @staticmethod
    def _sanitize_value(text: str) -> str:
        value = text.strip()
        # Gemma echoes the source before its answer: `source` -> "opt A" or "opt B".
        # The arrow is a safe cut point only when the echoed part looks like the
        # source (backticks or CJK), so a caption that legitimately contains an
        # arrow survives unchanged.
        arrows = [i for i in (value.find("->"), value.find("→"), value.find("=>")) if i != -1]
        if arrows:
            cut = max(arrows)
            if "`" in value[:cut] or re.search(r"[\u3400-\u9fff]", value[:cut]):
                value = value[cut:].lstrip("-=→> \t").strip()
        if len(value) >= 2 and value.startswith("`") and value.endswith("`"):
            value = value[1:-1].strip()
        value = re.sub(r"^[`\s]+", "", value)
        value = re.sub(r"[`.\s]+$", "", value)
        # Quoted alternative lists and "(Same as N)" notes keep only the first
        # real translation; quoted speech inside an otherwise bare sentence
        # ("Anh nói "đi" rồi bỏ đi.") matches neither pattern and is preserved.
        alternatives = re.match(
            r'^(?:\([^)]{1,40}\)\s*)?["“]([^"”]{1,200})["”]\s*(?:or|hoặc|and)\s*["“]', value
        )
        if alternatives:
            return alternatives.group(1).strip()
        single = re.match(r'^(?:\([^)]{1,40}\)\s*)?["“]([^"”]{1,200})["”]\s*[.,]?\s*$', value)
        if single:
            return single.group(1).strip()
        return value

    @classmethod
    def _parse_numbered_lines(cls, content: str, original_texts: List[str]) -> List[str]:
        translated: Dict[int, str] = {}
        for raw_line in content.splitlines():
            # Strip bullet markers and wrapping backticks that chatty models put
            # around the final "i: answer" line.
            line = re.sub(r"^[\*\s`\-]+", "", raw_line.strip())
            # Tolerate "N:", "N." and bullet prefixes from chatty models.
            match = re.match(r"^(\d{1,3})\s*[.:)\-]\s*(.*)$", line)
            if not match:
                continue
            # One physical line can carry several entries glued together
            # ("0: A`0: B", "0: A.0: B"). An embedded marker repeating the line
            # index or jumping forward starts the next entry; timestamps such
            # as "0:00" (digit after the separator) and parenthetical notes
            # like "(Same as 0)" (")" separator) do not.
            start_index = int(match.group(1))
            rest = match.group(2)
            cuts = [
                m for m in re.finditer(r"(?<!\d)(\d{1,3})\s*[.:]\s*(?!\d)", rest)
                if int(m.group(1)) >= start_index
            ]
            bounds = [m.start() for m in cuts] + [len(rest)]
            segments: List[tuple] = [(start_index, rest[: bounds[0]])]
            for k, m in enumerate(cuts):
                segments.append((int(m.group(1)), rest[m.end() : bounds[k + 1]]))
            # The first non-empty segment per index wins inside the line: the
            # trailing copies are re-echoes that the line end often truncates.
            line_translated: Dict[int, str] = {}
            for index, text in segments:
                text = re.sub(r"^\d+\s*[.:)\-]\s*", "", text.strip()).strip()
                text = cls._sanitize_value(text)
                if text and index not in line_translated:
                    line_translated[index] = text
            # Last wins ACROSS lines: reasoning responses echo the sources
            # first and place the final answers at the very end, so the last
            # occurrence of each index is the one to keep. Within a line the
            # first segment still wins (trailing re-echoes get line-end
            # truncated).
            for index, text in line_translated.items():
                translated[index] = text
        if len(translated) < len(original_texts):
            # Chain-of-thought responses bury the final answer mid-prose
            # ("...Is the format correct? Yes.0: Ánh mắt dịu dàng") where the
            # line-anchored regex cannot reach it. Collect candidates per index
            # with last-wins (the final answer trails the reasoning), then fill
            # strictly-empty slots so already-parsed answers stand. Out-of-range
            # indices are dropped by the fill loop.
            salvaged: Dict[int, str] = {}
            for raw_line in content.splitlines():
                match = re.search(r"(?<!\d)(\d{1,3})\s*[.:)\-]\s*(?!\d)(.+)$", raw_line.strip())
                if not match:
                    continue
                text = cls._sanitize_value(match.group(2).strip().strip("`").strip())
                if text:
                    salvaged[int(match.group(1))] = text
            for index in range(len(original_texts)):
                if not translated.get(index) and salvaged.get(index):
                    translated[index] = salvaged[index]
        # An unparsed line must NOT fall back to the source text: that rendered
        # original-language captions (with literal "\n") as if they were
        # translations. An empty value lets the router validation fail the
        # chunk so the fail-safe pass suppresses the source pixels instead.
        return [translated.get(index, "") for index in range(len(original_texts))]
