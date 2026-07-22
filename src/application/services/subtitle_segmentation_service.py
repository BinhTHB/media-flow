import re
from src.application.services.subtitle_text_formatter import SubtitleTextFormatter
from src.domain.entities.segment import Segment
from src.domain.entities.subtitle_quality_config import SubtitleQualityConfig

class SubtitleSegmentationService:
    def __init__(self, config: SubtitleQualityConfig | None = None, enabled: bool = True,
                 prefer_sentence_boundary: bool = True, prefer_punctuation: bool = True,
                 proportional_timing: bool = True):
        self.config = config or SubtitleQualityConfig()
        self.enabled = enabled
        self.prefer_sentence_boundary = prefer_sentence_boundary
        self.prefer_punctuation = prefer_punctuation
        self.proportional_timing = proportional_timing

    def segment(self, segments: list[Segment], is_translated: bool = False) -> list[Segment]:
        if not self.enabled:
            return segments
        result = []
        for segment in segments:
            result.extend(self._split_segment(segment, is_translated))
        return result

    def _active_text(self, segment: Segment, is_translated: bool) -> str:
        return segment.translated_text if is_translated else segment.text

    def _set_active_text(self, segment: Segment, value: str, is_translated: bool) -> None:
        if is_translated:
            segment.translated_text = value
        else:
            segment.text = value

    def _split_segment(self, segment: Segment, is_translated: bool) -> list[Segment]:
        text = self._active_text(segment, is_translated).strip()
        if not text:
            return []
        chunks = self._chunk_text(text)
        if len(chunks) <= 1:
            formatted = SubtitleTextFormatter.format_text(text, self.config.max_chars_per_line, self.config.max_lines_per_subtitle)
            self._set_active_text(segment, formatted, is_translated)
            return [segment]

        total_duration = max(segment.end - segment.start, 0.0)
        total_chars = max(sum(len(c) for c in chunks), 1)
        cursor = segment.start
        out = []
        for idx, chunk in enumerate(chunks):
            if idx == len(chunks) - 1:
                end = segment.end
            elif self.proportional_timing:
                portion = len(chunk) / total_chars
                duration = total_duration * portion
                duration = max(duration, self.config.min_duration if total_duration >= self.config.min_duration else duration)
                duration = min(duration, self.config.max_duration)
                remaining = len(chunks) - idx - 1
                reserved_duration = remaining * self.config.min_duration
                max_end = segment.end - reserved_duration
                end = min(segment.end, cursor + duration)
                if total_duration >= len(chunks) * self.config.min_duration:
                    end = min(end, max_end)
            else:
                end = min(segment.end, cursor + min(self.config.max_duration, max(self.config.min_duration, total_duration / len(chunks))))
            if end <= cursor and segment.end > cursor:
                end = segment.end
            new_segment = Segment(
                id=f"{segment.id}_{idx}",
                start=cursor,
                end=end,
                text=segment.text,
                translated_text=segment.translated_text,
                source=segment.source,
                confidence=segment.confidence,
                confidence_available=segment.confidence_available,
                bbox=segment.bbox,
                metadata=dict(segment.metadata),
            )
            formatted = SubtitleTextFormatter.format_text(chunk, self.config.max_chars_per_line, self.config.max_lines_per_subtitle)
            self._set_active_text(new_segment, formatted, is_translated)
            out.append(new_segment)
            cursor = end
        return out

    def _fits_subtitle_shape(self, text: str) -> bool:
        lines = SubtitleTextFormatter.wrap_lines(text, self.config.max_chars_per_line)
        return len(lines) <= self.config.max_lines_per_subtitle

    def _chunk_text(self, text: str) -> list[str]:
        if self._fits_subtitle_shape(text):
            return [text]
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) > 1:
            chunks = []
            for sentence in sentences:
                chunks.extend(self._chunk_words(sentence))
            return self._improve_boundaries(chunks)
        return self._improve_boundaries(self._chunk_words(text))

    def _chunk_words(self, text: str) -> list[str]:
        tokens = text.split()
        chunks = []
        current = []
        for token in tokens:
            proposed = " ".join(current + [token])
            if current and not self._fits_subtitle_shape(proposed):
                chunks.append(" ".join(current))
                current = [token]
            else:
                current.append(token)
        if current:
            chunks.append(" ".join(current))
        balanced = self._partition_balanced(tokens, len(chunks))
        return balanced or chunks

    def _partition_balanced(self, tokens: list[str], chunk_count: int) -> list[str] | None:
        if chunk_count == 1:
            text = " ".join(tokens)
            return [text] if self._fits_subtitle_shape(text) else None

        best = None
        best_max_length = None
        for index in range(1, len(tokens) - chunk_count + 2):
            head = " ".join(tokens[:index])
            if not self._fits_subtitle_shape(head):
                break
            tail = self._partition_balanced(tokens[index:], chunk_count - 1)
            if tail is None:
                continue
            candidate = [head, *tail]
            max_length = max(len(chunk) for chunk in candidate)
            if best_max_length is None or max_length < best_max_length:
                best = candidate
                best_max_length = max_length
        return best

    def _improve_boundaries(self, chunks: list[str]) -> list[str]:
        continuations = (
            "tôi đã",
            "anh ta không",
            "không",
            "anh",
            "và",
            "sẽ",
            "rằng",
        )
        result = [c.strip() for c in chunks if c.strip()]
        for index in range(len(result) - 1):
            lower = result[index].lower()
            phrase = next((item for item in continuations if lower.endswith(item)), "")
            if phrase:
                result[index] = result[index][:-len(phrase)].rstrip(" ,")
                result[index + 1] = f"{phrase.capitalize()} {result[index + 1]}"
        return [chunk for chunk in result if chunk]
