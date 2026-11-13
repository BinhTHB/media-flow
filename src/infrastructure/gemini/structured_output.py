from __future__ import annotations

import json
import re
from typing import Any, Callable, Optional, TypeVar

from src.infrastructure.gemini.exceptions import GeminiValidationError

T = TypeVar("T")

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_OBJ_RE = re.compile(r"[\{\[]")


def extract_json_text(text: str) -> str:
    if not text or not text.strip():
        raise GeminiValidationError("Empty model output; cannot parse structured JSON.")
    stripped = text.strip()
    fence = _FENCE_RE.search(stripped)
    if fence:
        stripped = fence.group(1).strip()
    return stripped


def parse_json(text: str) -> Any:
    cleaned = extract_json_text(text)
    decoder = json.JSONDecoder()
    match = _OBJ_RE.search(cleaned)
    if not match:
        raise GeminiValidationError("No JSON object or array found in model output.")
    try:
        obj, _ = decoder.raw_decode(cleaned, match.start())
    except json.JSONDecodeError as exc:
        raise GeminiValidationError(f"Failed to decode JSON from model output: {exc}") from exc
    return obj


def parse_json_recovering(text: str) -> Any:
    """Parse model JSON, keeping complete values when the output was cut off.

    A long transcript can end mid-string once the response hits its limit.
    Segments that already closed are still usable.
    """
    cleaned = extract_json_text(text)
    decoder = json.JSONDecoder()
    match = _OBJ_RE.search(cleaned)
    if not match:
        raise GeminiValidationError("No JSON object or array found in model output.")
    fragment = cleaned[match.start():]
    try:
        obj, _ = decoder.raw_decode(fragment)
        return obj
    except json.JSONDecodeError as exc:
        # A missing comma or a cut string makes the tail unusable. Values that
        # closed before that point are still a transcript.
        cutoff = exc.pos if isinstance(exc.pos, int) and exc.pos > 0 else len(fragment)
        end = _last_complete_value_end(fragment[:cutoff])
        if end < 0:
            raise GeminiValidationError(
                "Failed to decode JSON from model output: truncated with no complete value"
            )
        head = fragment[: end + 1].rstrip().rstrip(",")
        repaired = head + _closers_for(head)
        try:
            obj, _ = decoder.raw_decode(repaired)
        except json.JSONDecodeError as exc:
            raise GeminiValidationError(f"Failed to decode JSON from model output: {exc}") from exc
        return obj


def _last_complete_value_end(fragment: str) -> int:
    in_string = False
    escaped = False
    depth = 0
    last = -1
    for index, char in enumerate(fragment):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
            if depth >= 1:
                last = index
    return last


def _closers_for(head: str) -> str:
    stack: list[str] = []
    in_string = False
    escaped = False
    for char in head:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in "}]" and stack:
            stack.pop()
    return "".join(reversed(stack))


def parse_structured(text: str, factory: Optional[Callable[[Any], T]] = None) -> T:
    obj = parse_json(text)
    if factory is None:
        return obj  # type: ignore[return-value]
    try:
        return factory(obj)
    except Exception as exc:
        raise GeminiValidationError(f"Structured output failed factory validation: {exc}") from exc
