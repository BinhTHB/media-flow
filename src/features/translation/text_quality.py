"""Language and caption checks shared by subtitle replacement and dubbing.

Two problems this solves, both observed on real re-upload sources:

1. **Source already in the target language.** A Vietnamese short re-uploaded
   with ``--target-language vi`` used to be sent to the translator anyway, which
   produced diacritic shuffles ("Kì lạ" -> "Kỳ lạ") or invented text. Translation
   must be skipped when the source already is the target.

2. **OCR noise promoted to captions.** Detectors label watermark counters and
   on-screen numbers as ``subtitle``; OCR then reads "122" and the renderer
   paints it over the video. A caption has letters; a counter does not.

Dubbing uses the same language check: a line that comes back still in the
source language is not spoken.

Everything here is pure and dependency-free so it can be unit-tested without
media.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

#: Vietnamese-specific letters. These do not occur in other Latin-script
#: languages, so even one of them is strong evidence.
_VIETNAMESE_MARKS = set(
    "ăâđêôơư"
    "ĂÂĐÊÔƠƯ"
    "àảãáạằẳẵắặầẩẫấậèẻẽéẹềểễếệìỉĩíịòỏõóọồổỗốộờởỡớợùủũúụừửữứựỳỷỹýỵ"
    "ÀẢÃÁẠẰẲẴẮẶẦẨẪẤẬÈẺẼÉẸỀỂỄẾỆÌỈĨÍỊÒỎÕÓỌỒỔỖỐỘỜỞỠỚỢÙỦŨÚỤỪỬỮỨỰỲỶỸÝỴ"
)

#: Characters used to detect the writing system of a string.
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
_LATIN = re.compile(r"[A-Za-zÀ-ỹ]")
_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)
_DIGIT = re.compile(r"\d")


def detect_script(text: str) -> str:
    """Coarse writing-system label for a caption string.

    Returns one of ``cjk``, ``latin``, ``mixed``, ``numeric``, ``empty``.
    """
    value = (text or "").strip()
    if not value:
        return "empty"
    cjk = len(_CJK.findall(value))
    latin = len(_LATIN.findall(value))
    digits = len(_DIGIT.findall(value))
    if cjk and latin:
        return "mixed"
    if cjk:
        return "cjk"
    if latin:
        return "latin"
    if digits:
        return "numeric"
    return "empty"


def looks_vietnamese(text: str) -> bool:
    """True when the text is confidently Vietnamese.

    Requires at least one Vietnamese-specific letter OR a majority of
    Vietnamese-legal diacritics. Plain ASCII ("OK", "2024") is deliberately
    *not* classified as Vietnamese: it carries no evidence either way, and
    guessing "already Vietnamese" from ASCII would skip legitimate translation
    of English captions.
    """
    value = (text or "").strip()
    if not value:
        return False
    if any(ch in _VIETNAMESE_MARKS for ch in value):
        return True
    decomposed = unicodedata.normalize("NFD", value)
    combining = sum(1 for ch in decomposed if unicodedata.combining(ch))
    return combining >= max(2, len(value) // 6)


def is_already_target_language(text: str, target_language: str) -> bool:
    """True when ``text`` is plausibly already written in ``target_language``.

    Only Vietnamese is detected positively; other targets fall back to False so
    behaviour stays "translate" rather than silently skipping work.
    """
    target = (target_language or "").strip().lower()
    if target in {"vi", "vi-vn", "vietnamese", "tiếng việt"}:
        return looks_vietnamese(text)
    return False


def is_caption_worthy(text: str, *, min_letters: int = 2) -> bool:
    """True when OCR output is plausibly a caption rather than noise.

    Rejects counters ("122"), lone punctuation, and single-character OCR
    fragments. A real caption almost always contains at least two letters.
    """
    value = (text or "").strip()
    if not value:
        return False
    letters = _LETTER.findall(value)
    if len(letters) < min_letters:
        return False
    # A string that is mostly digits ("ab 1234") is a watermark counter, not
    # a caption. A brand with a short number ("CCTV 12") still has more
    # letters than digits and stays a caption.
    digits = len(_DIGIT.findall(value))
    if digits and digits >= len(letters) * 2:
        return False
    return True


def normalize_for_compare(text: str) -> str:
    """Casefold + strip diacritics, for "is this the same caption" checks."""
    value = (text or "").strip().casefold()
    decomposed = unicodedata.normalize("NFD", value)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", stripped).strip()


def is_effectively_identical(a: str, b: str) -> bool:
    """True when two strings differ only in diacritics, case, or punctuation.

    Used to detect a "translation" that returned the source unchanged, which
    is the observable symptom of translating into the source language.
    """
    left = normalize_for_compare(a)
    right = normalize_for_compare(b)
    if not left or not right:
        return False
    if left == right:
        return True
    # Punctuation-only differences ("Kỳ lạ!" vs "Kỳ lạ") also count.
    return re.sub(r"[^\w\s]", "", left) == re.sub(r"[^\w\s]", "", right)


def pick_source_language(text: str) -> Optional[str]:
    """Best-effort language hint for a caption, or None when unsure."""
    script = detect_script(text)
    if script == "cjk":
        return "zh"
    if script == "latin":
        return "vi" if looks_vietnamese(text) else "en"
    return None


def accept_spoken_line(source: str, candidate: str, target_language: str) -> str:
    """Text safe to dub, or empty when the line is not target-language speech.

    A source that is already the target language is spoken unchanged.
    Re-translating it shuffles diacritics. A candidate that merely echoes a
    foreign source, or that is still CJK, is rejected so the dub does not
    speak the original language.
    """
    source = (source or "").strip()
    candidate = (candidate or "").strip()
    if is_already_target_language(source, target_language):
        # A shorter Vietnamese rewrite (duration fit, manual edit) is kept.
        # A diacritic-only shuffle of the source is not: speak the original.
        if (
            candidate
            and is_already_target_language(candidate, target_language)
            and not is_effectively_identical(source, candidate)
        ):
            return candidate
        return source
    if not candidate:
        return ""
    if is_effectively_identical(source, candidate):
        return ""
    if detect_script(candidate) == "cjk":
        return ""
    return candidate
