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


def is_collapsed_speech(text: str) -> bool:
    """True when ASR dropped most vowels and left consonant crumbs.

    A real sentence, including Vietnamese written without diacritics, still
    has a vowel in nearly every word. A collapsed Whisper window looks like
    ``b n Ngay gi c v h l``.
    """
    value = (text or "").strip()
    letters = [ch for ch in value if ch.isalpha()]
    if len(letters) < 24:
        return False
    vowels = sum(1 for ch in letters if _is_vowel_letter(ch))
    if vowels / len(letters) >= 0.28:
        return False
    tokens = re.findall(r"[^\W\d_]+", value)
    if len(tokens) < 8:
        return False
    voiced = sum(1 for token in tokens if any(_is_vowel_letter(ch) for ch in token))
    return voiced / len(tokens) < 0.45


# Real syllables ending in ữ/ũ + ng. "những" and "cũng" are common;
# "bững" is not a syllable, it is an ASR miss for "vững".
_UNG_OK = frozenset(
    {
        "bùng", "chùng", "chúng", "cùng", "cũng", "dũng", "dùng", "dụng",
        "hùng", "hững", "khùng", "khững", "lũng", "lững", "mừng", "ngừng",
        "những", "phùng", "phững", "rùng", "rừng", "sừng", "sũng", "thùng",
        "thũng", "trũng", "từng", "vững", "vũng", "xứng", "xùng", "đúng",
    }
)
_ODD_UNG = re.compile(r"[0-9A-Za-zÀ-ỹ]*[ữũ]ng", re.IGNORECASE)
_ASCII_WORD = re.compile(r"\b[A-Za-z]{4,}\b")
_VI_SYLLABLE = re.compile(
    r"^(?:ngh|gh|gi|kh|ng|nh|ph|qu|th|tr|ch|[bcdghklmnpqrstvx])?"
    r"(?:oai|oay|uai|uay|ieu|uya|uyu|yeu|ai|ao|au|ay|eo|ia|iu|oa|oe|oi|oo|ua|ui|uy|a|e|i|o|u|y)"
    r"(?:ch|nh|ng|[cmnpt])?$",
    re.IGNORECASE,
)
_SHORT_ASCII = re.compile(r"(?<![A-Za-zÀ-ỹ])([A-Za-z]{1,2})(?![A-Za-zÀ-ỹ])")
_LOANWORDS = frozenset(
    {
        "android", "burger", "covid", "douyin", "email", "facebook", "google",
        "internet", "iphone", "online", "pizza", "radio", "smartphone", "studio",
        "tiktok", "video", "virus", "website", "wifi", "youtube",
    }
)
_SHORT_ASCII_OK = frozenset(
    {
        "ai", "anh", "ba", "bo", "chi", "co", "con", "da", "de", "di", "em", "eu",
        "hay", "ho", "khi", "la", "ma", "me", "moi", "mot", "nam", "no", "o", "ok",
        "ong", "ra", "se", "ta", "thi", "toi", "tu", "tv", "uk", "us", "va", "ve",
        "vi", "vn", "xe",
    }
)


def speech_defect_tokens(text: str) -> list[str]:
    """Tokens that cannot be ordinary Vietnamese speech.

    A line can keep its vowels and still be an ASR miss: ``bánh tisa``,
    ``bững vàng``, ``từng re biểu``. Names and loanwords that the second
    transcript also contains are left alone by the repair step.
    """
    value = text or ""
    defects: list[str] = []
    seen: set[str] = set()

    def add(token: str) -> None:
        key = token.casefold()
        if key in seen:
            return
        seen.add(key)
        defects.append(token)

    vietnamese_line = any(ch in _VIETNAMESE_MARKS for ch in value)
    for token in _ASCII_WORD.findall(value):
        if (
            vietnamese_line
            and token.casefold() not in _LOANWORDS
            and _VI_SYLLABLE.fullmatch(token) is None
        ):
            add(token)
    if vietnamese_line:
        for token in _SHORT_ASCII.findall(value):
            if token.casefold() not in _SHORT_ASCII_OK:
                add(token)
    for match in _ODD_UNG.finditer(value):
        token = match.group(0)
        if token.casefold() not in _UNG_OK:
            add(token)
    return defects


def substitute_speech_defects(text: str, evidence: str) -> str:
    """Swap one broken token for the word the second transcript actually heard.

    The second read is often a whole paragraph. Pasting that paragraph into a
    two-second slot covers the picture. This only exchanges ``tisa`` for
    ``pizza``, or ``bững`` for ``vững``, when the evidence names that word.
    """
    value = text or ""
    proof = evidence or ""
    if not value.strip() or not proof.strip() or is_collapsed_speech(value):
        return value
    defects = speech_defect_tokens(value)
    if not defects:
        return value

    updated = value
    proof_cf = proof.casefold()
    ascii_defects = [
        token
        for token in defects
        if _ASCII_WORD.fullmatch(token) and _PROPER_NAME.fullmatch(token) is None
    ]
    loans = [
        word
        for word in sorted(_LOANWORDS)
        if re.search(rf"\b{re.escape(word)}\b", proof_cf)
        and re.search(rf"\b{re.escape(word)}\b", value.casefold()) is None
    ]
    if len(ascii_defects) == 1 and len(loans) == 1:
        updated = re.sub(
            rf"\b{re.escape(ascii_defects[0])}\b",
            loans[0],
            updated,
            count=1,
        )

    ung_defects = [
        token for token in defects if _ODD_UNG.fullmatch(token) and token.casefold() not in _UNG_OK
    ]
    if len(ung_defects) == 1:
        seen: set[str] = set()
        candidates: list[str] = []
        for token in re.findall(r"[^\W\d_]+", proof):
            key = token.casefold()
            if key in _UNG_OK and key not in seen:
                seen.add(key)
                candidates.append(token)
        replacement = _closest_token(ung_defects[0], candidates, max_distance=1)
        if replacement is not None:
            updated = re.sub(
                rf"\b{re.escape(ung_defects[0])}\b",
                replacement,
                updated,
                count=1,
                flags=re.IGNORECASE,
            )
    return updated


_PROPER_NAME = re.compile(r"\b[A-Z][a-z]{2,}\b")


def _closest_token(source: str, candidates: list[str], max_distance: int) -> Optional[str]:
    ranked = sorted(
        ((_edit_distance(source.casefold(), token.casefold()), token) for token in candidates),
        key=lambda item: item[0],
    )
    if not ranked or ranked[0][0] > max_distance:
        return None
    if len(ranked) > 1 and ranked[1][0] == ranked[0][0]:
        return None
    return ranked[0][1]


def _edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for index, left_ch in enumerate(left, start=1):
        current = [index]
        for jndex, right_ch in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[jndex] + 1,
                    previous[jndex - 1] + (left_ch != right_ch),
                )
            )
        previous = current
    return previous[-1]


def repair_asr_phrases(text: str) -> str:
    """Correct a few ASR phrase-splits that are not real Vietnamese.

    ``rẻ như cho`` stays. ``rẻ biểu``, ``rèm bỉu`` and ``dèm bỉu`` are the same
    missed idiom, ``dè bỉu``. An impossible syllable such as ``bững vàng``
    takes the legal syllable that keeps that vowel and the next word's onset.
    """
    value = text or ""
    for pattern, replacement in _ASR_PHRASES:
        value = pattern.sub(replacement, value)
    return _repair_odd_ung(value)


def _repair_odd_ung(text: str) -> str:
    tokens = list(_TOKEN.finditer(text))
    updated = text
    for index, match in enumerate(tokens):
        token = match.group(0)
        if token.casefold() in _UNG_OK or _ODD_UNG.fullmatch(token) is None:
            continue
        vowel_match = re.search(r"[ữũ]", token.casefold())
        if vowel_match is None:
            continue
        vowel = vowel_match.group(0)
        same_vowel = [
            word
            for word in _UNG_OK
            if vowel in word and _edit_distance(token.casefold(), word) == 1
        ]
        following = tokens[index + 1].group(0) if index + 1 < len(tokens) else ""
        onset = following[:1].casefold()
        matched = [word for word in same_vowel if word[:1] == onset]
        if len(matched) != 1:
            continue
        updated = re.sub(
            rf"\b{re.escape(token)}\b",
            matched[0],
            updated,
            count=1,
            flags=re.IGNORECASE,
        )
    return updated


def restore_introduced_names(texts: list[str]) -> list[str]:
    """Put a name back where a later line spells it with Vietnamese tones.

    ``Ông tên Mehmet`` makes a later ``mê mét`` the same name. A pair with no
    introduced name within one letter stays as written.
    """
    names: list[str] = []
    seen: set[str] = set()
    for text in texts:
        for name in _PROPER_NAME.findall(text or ""):
            key = name.casefold()
            if key in seen:
                continue
            seen.add(key)
            names.append(name)
    if not names:
        return [text or "" for text in texts]
    restored: list[str] = []
    for text in texts:
        updated = text or ""
        spans = list(_TOKEN.finditer(updated))
        pieces: list[tuple[int, int, str]] = []
        index = 0
        while index < len(spans) - 1:
            left, right = spans[index], spans[index + 1]
            if right.start() - left.end() > 2:
                index += 1
                continue
            pair = left.group() + right.group()
            if not any(ch in _VIETNAMESE_MARKS for ch in pair):
                index += 1
                continue
            joined = _latin_key(pair)
            best_name = ""
            best_distance = 99
            for name in names:
                distance = _edit_distance(joined, _latin_key(name))
                if distance < best_distance:
                    best_distance = distance
                    best_name = name
            if best_name and best_distance <= 1:
                pieces.append((left.start(), right.end(), best_name))
                index += 2
                continue
            index += 1
        if not pieces:
            restored.append(updated)
            continue
        cursor = 0
        chunks: list[str] = []
        for start, end, name in pieces:
            chunks.append(updated[cursor:start])
            chunks.append(name)
            cursor = end
        chunks.append(updated[cursor:])
        restored.append("".join(chunks))
    return restored


def _latin_key(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).casefold()


_TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)


_ASR_PHRASES = (
    (re.compile(r"\brẻ biểu\b", re.IGNORECASE), "dè bỉu"),
    (re.compile(r"\brèm bỉu\b", re.IGNORECASE), "dè bỉu"),
    (re.compile(r"\bdèm bỉu\b", re.IGNORECASE), "dè bỉu"),
    (re.compile(r"\bre biểu\b", re.IGNORECASE), "dè bỉu"),
    (re.compile(r"\bbánh tisa\b", re.IGNORECASE), "bánh pizza"),
    (re.compile(r"\bbỏ ngoài tay\b", re.IGNORECASE), "bỏ ngoài tai"),
)


def _is_vowel_letter(ch: str) -> bool:
    base = unicodedata.normalize("NFD", ch)
    return bool(base) and base[0].casefold() in "aeiouy"


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


_VI_LANGUAGE_LABELS = frozenset(
    {"vi", "vi-vn", "vietnamese", "tiếng việt", "tieng viet"}
)


def is_already_target_language(text: str, target_language: str) -> bool:
    """True when ``text`` is plausibly already written in ``target_language``.

    Only Vietnamese is detected positively; other targets fall back to False so
    behaviour stays "translate" rather than silently skipping work.
    """
    target = (target_language or "").strip().lower()
    if target in _VI_LANGUAGE_LABELS:
        return looks_vietnamese(text)
    return False


def detected_language_is_target(detected: str, target_language: str) -> bool:
    """True when an ASR/job language label already matches ``target_language``.

    Text heuristics miss undiacritic Vietnamese ("tranh nhau."). The recognizer
    language is a stronger signal than letter shapes for those lines.
    """
    target = (target_language or "").strip().lower()
    detected = (detected or "").strip().lower()
    if not detected or detected in {"auto", "unknown"}:
        return False
    if target in _VI_LANGUAGE_LABELS:
        return detected in _VI_LANGUAGE_LABELS
    return False


def should_skip_spoken_translation(
    text: str,
    target_language: str,
    *,
    detected_language: str = "auto",
    job_source_language: str = "auto",
) -> bool:
    """True when a spoken line must keep its wording and not call a translator.

    CJK is never skipped from a language label alone: a Chinese caption in a
    Vietnamese-narrated video still needs translation.
    """
    if is_already_target_language(text, target_language):
        return True
    if detect_script(text) == "cjk":
        return False
    return detected_language_is_target(
        detected_language, target_language
    ) or detected_language_is_target(job_source_language, target_language)


def is_dangling_fragment(
    text: str,
    *,
    box_width: float,
    frame_width: float,
    duration: float,
) -> bool:
    """True when a tiny chip is not a caption worth translating.

    Two characters in a box under 7% of the frame, on screen for under
    1.2s, are a connective scrap such as 那是. A name on the costume is
    wider or stays longer, so it still translates.
    """
    value = (text or "").strip()
    cjk = len(_CJK.findall(value))
    if cjk == 0 or cjk > 2:
        return False
    if duration > 1.2:
        return False
    return box_width < 0.07 * max(1.0, frame_width)


_CREDIT_MARK = re.compile(r"^[※＊©®]")


def is_production_credit(text: str) -> bool:
    """True when burned-in text is a credit or disclaimer, not dialogue.

    A line that opens with a reference mark (※人形です) or a short maker
    tag (西瓜君AI制作) is removed, not translated onto the picture.
    A spoken caption does not start with that mark and is longer than a tag.
    """
    value = re.sub(r"\s+", "", (text or "").strip())
    if not value:
        return False
    if _CREDIT_MARK.match(value):
        return True
    cjk = _CJK.findall(value)
    return value.endswith("制作") and 0 < len(cjk) <= 12


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


def accept_spoken_line(
    source: str,
    candidate: str,
    target_language: str,
    *,
    detected_language: str = "auto",
    job_source_language: str = "auto",
) -> str:
    """Text safe to dub, or empty when the line is not target-language speech.

    A source that is already the target language is spoken unchanged.
    Re-translating it shuffles diacritics. A candidate that merely echoes a
    foreign source, or that is still CJK, is rejected so the dub does not
    speak the original language. ASR language labels cover undiacritic
    Vietnamese that letter-shape checks miss.
    """
    source = (source or "").strip()
    candidate = (candidate or "").strip()
    if is_collapsed_speech(source):
        return ""
    if should_skip_spoken_translation(
        source,
        target_language,
        detected_language=detected_language,
        job_source_language=job_source_language,
    ):
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
