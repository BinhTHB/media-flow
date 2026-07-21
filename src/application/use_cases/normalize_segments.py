from typing import Callable, List
from src.domain.entities.segment import Segment
from src.domain.entities.subtitle_quality_config import SubtitleQualityConfig
from src.application.services.text_similarity import TextSimilarityService
from src.application.services.subtitle_text_formatter import SubtitleTextFormatter
from src.application.services.content_quality_auditor import ContentQualityAuditor

class NormalizeSegmentsUseCase:
    def __init__(self, quality_config: SubtitleQualityConfig = None):
        self.config = quality_config or SubtitleQualityConfig()
        self.metrics = {}

    def execute(
        self,
        segments: List[Segment],
        is_translated: bool = False,
        video_duration: float | None = None,
        replacement_required: bool = False,
    ) -> List[Segment]:
        self.metrics = {}
        if not segments:
            return []

        def get_text(s: Segment) -> str:
            return s.translated_text if is_translated else s.text

        def set_text(s: Segment, val: str) -> None:
            if is_translated:
                s.translated_text = val
            else:
                s.text = val

        # 1. Trim whitespace
        if self.config.trim_whitespace:
            for s in segments:
                set_text(s, get_text(s).strip())
                s.text = s.text.strip()
                s.translated_text = s.translated_text.strip()

        # 2. Remove empty segments
        if self.config.remove_empty_segments:
            segments = [s for s in segments if get_text(s)]

        # 3. Sort by timestamp (always)
        segments.sort(key=lambda s: s.start)

        # Record counts before normalization
        speech_before = [s for s in segments if s.source == "speech"]
        ocr_before = [s for s in segments if s.source == "ocr"]
        self.metrics = {
            "speech_before_count": len(speech_before),
            "ocr_before_count": len(ocr_before),
            "speech_before_normalization": len(speech_before),
            "ocr_before_normalization": len(ocr_before),
            "long_speech_segments_detected": 0,
            "long_speech_segments_split_or_queued": 0,
            "long_speech_segments_dropped": 0,
            "suppressed_ocr_fragments": [],
            "redundant_ocr_count": 0,
        }

        self._split_long_speech(segments, get_text, is_translated, video_duration)
        if not replacement_required:
            segments = self._suppress_redundant_ocr(segments, get_text)

        # 4. Prefer speech over OCR when timestamps overlap and text is similar
        if self.config.prefer_speech_over_ocr and not replacement_required:
            filtered = []
            for s in segments:
                keep = True
                if s.source == "ocr":
                    for other in segments:
                        if other.source == "speech":
                            overlap = max(s.start, other.start) < min(s.end, other.end)
                            if overlap:
                                similarity = TextSimilarityService.get_similarity(get_text(s), get_text(other))
                                if similarity > 0.6:
                                    keep = False
                                    break
                if keep:
                    filtered.append(s)
            segments = filtered

        # 5. Deduplicate repeated text within each timeline.
        if self.config.deduplicate:
            filtered = []
            for s in segments:
                if not filtered:
                    filtered.append(s)
                    continue
                last = filtered[-1]
                if replacement_required and last.source != s.source:
                    filtered.append(s)
                    continue
                similarity = TextSimilarityService.get_similarity(get_text(s), get_text(last))
                if similarity <= 0.8:
                    filtered.append(s)
            segments = filtered

        # 6. Merge very short neighboring segments when safe
        if self.config.merge_gap_threshold > 0:
            merged = []
            for s in segments:
                if not merged:
                    merged.append(s)
                else:
                    last = merged[-1]
                    gap = s.start - last.end
                    if (last.source == s.source and 
                        0 <= gap < self.config.merge_gap_threshold and 
                        (s.end - last.start) <= self.config.max_duration):
                        set_text(last, get_text(last) + " " + get_text(s))
                        if is_translated:
                            last.text = last.text + " " + s.text
                        else:
                            last.translated_text = last.translated_text + " " + s.translated_text
                        last.end = s.end
                        last.confidence = (last.confidence + s.confidence) / 2.0
                        last.confidence_available = (
                            last.confidence_available and s.confidence_available
                        )
                        boxes = [box for box in (last.bbox, s.bbox) if box is not None]
                        if boxes:
                            last.bbox = {
                                "left": min(box["left"] for box in boxes),
                                "top": min(box["top"] for box in boxes),
                                "right": max(box["right"] for box in boxes),
                                "bottom": max(box["bottom"] for box in boxes),
                            }
                        merged_ids = getattr(last, "_merged_source_ids", [last.id])
                        setattr(
                            last,
                            "_merged_source_ids",
                            merged_ids + getattr(s, "_merged_source_ids", [s.id]),
                        )
                    else:
                        merged.append(s)
            segments = merged

        # 7. Clamp to the video timeline and select one readable caption per interval.
        normalized = []
        for s in segments:
            s.start = max(s.start, 0.0)
            s.end = min(s.end, video_duration) if video_duration is not None else s.end
            if s.end <= s.start:
                continue
            if not (replacement_required and s.source == "ocr"):
                # Replacement mode: OCR display segments must cover the source
                # subtitle for its full on-screen span; never clamp to max_duration.
                s.end = min(s.end, s.start + self.config.max_duration)
            if s.end - s.start < self.config.min_duration:
                if video_duration is not None:
                    s.end = min(s.start + self.config.min_duration, video_duration)
                else:
                    s.end = s.start + self.config.min_duration
            if s.end - s.start >= self.config.min_duration:
                normalized.append(s)
        
        segments = self._resolve_overlaps(normalized, get_text, replacement_required)

        # Record counts and metrics after normalization
        speech_after = [s for s in segments if s.source == "speech"]
        ocr_after = [s for s in segments if s.source == "ocr"]
        
        represented_speech_ids = {
            source_id
            for segment in speech_after
            for source_id in getattr(segment, "_merged_source_ids", [segment.id])
        }
        dropped_speech_segments = [
            segment for segment in speech_before
            if segment.id not in represented_speech_ids
        ]
        dropped_speech = len(dropped_speech_segments)
        dropped_ocr = max(0, len(ocr_before) - len(ocr_after))
        
        total_duration = sum(s.end - s.start for s in speech_after)
        speech_ratio = total_duration / video_duration if video_duration and video_duration > 0 else 0.0
        
        drop_reasons = []
        if dropped_ocr > 0:
            drop_reasons.append("ocr_overlap_or_duplication")
        if dropped_speech > 0:
            drop_reasons.append("speech_overlap_under_min_duration")

        self.metrics.update({
            "speech_after_count": len(speech_after),
            "ocr_after_count": len(ocr_after),
            "speech_after_normalization": len(speech_after),
            "ocr_after_normalization": len(ocr_after),
            "dropped_speech_count": dropped_speech,
            "dropped_ocr_count": dropped_ocr,
            "drop_reasons": drop_reasons,
            "dropped_speech_reasons": (
                ["speech_overlap_under_min_duration"] if dropped_speech > 0 else []
            ),
            "speech_coverage_ratio": speech_ratio,
            "long_speech_segments_dropped": (
                self.metrics["long_speech_segments_detected"]
                - len({
                    getattr(segment, "_long_speech_parent_id", None)
                    for segment in speech_after
                    if getattr(segment, "_long_speech_parent_id", None)
                })
            ),
            "dropped_speech_segments": [
                {
                    "id": segment.id,
                    "text": get_text(segment),
                    "reason": "overlap_resolution",
                }
                for segment in dropped_speech_segments
            ],
            "final_speech_caption_count": len(speech_after),
            "final_ocr_caption_count": len(ocr_after),
        })

        # 8. Wrap long lines only, do not truncate (segmentation will handle splitting)
        for s in segments:
            wrapped_lines = SubtitleTextFormatter.wrap_lines(get_text(s), self.config.max_chars_per_line)
            set_text(s, "\n".join(wrapped_lines))

        return segments

    def _suppress_redundant_ocr(
        self,
        segments: List[Segment],
        get_text: Callable[[Segment], str],
    ) -> List[Segment]:
        speech_segments = [segment for segment in segments if segment.source == "speech"]
        kept = []
        suppressed = []
        for segment in segments:
            if segment.source != "ocr":
                kept.append(segment)
                continue

            text = get_text(segment)
            _, findings = ContentQualityAuditor.audit(text, "ocr")
            low_information = len(text.split()) <= 3
            nearby_speech = [
                speech for speech in speech_segments
                if max(segment.start, speech.start) < min(segment.end, speech.end)
            ]
            covered = any(
                self._token_coverage(text, get_text(speech)) >= 0.6
                for speech in nearby_speech
            )
            if covered and (findings or low_information):
                reasons = ["semantically_covered_by_speech"]
                if "incomplete_vietnamese_fragment" in findings:
                    reasons.append("incomplete_fragment")
                if low_information:
                    reasons.append("low_information")
                suppressed.append({
                    "id": segment.id,
                    "text": text,
                    "reasons": reasons,
                })
                continue
            kept.append(segment)

        self.metrics["suppressed_ocr_fragments"] = suppressed
        self.metrics["redundant_ocr_count"] = len(suppressed)
        return kept

    @staticmethod
    def _token_coverage(fragment: str, full_text: str) -> float:
        fragment_tokens = [
            token.strip(".,!?;:()[]{}\"'").lower()
            for token in fragment.split()
            if token.strip(".,!?;:()[]{}\"'")
        ]
        full_tokens = [
            token.strip(".,!?;:()[]{}\"'").lower()
            for token in full_text.split()
            if token.strip(".,!?;:()[]{}\"'")
        ]
        if not fragment_tokens or not full_tokens:
            return 0.0
        matched = sum(
            any(
                TextSimilarityService.get_similarity(token, candidate) >= 0.75
                for candidate in full_tokens
            )
            for token in fragment_tokens
        )
        return matched / len(fragment_tokens)

    def _split_long_speech(
        self,
        segments: List[Segment],
        get_text: Callable[[Segment], str],
        is_translated: bool,
        video_duration: float | None,
    ) -> None:
        ocr_segments = [segment for segment in segments if segment.source == "ocr"]
        replacements = {}
        for speech in (segment for segment in segments if segment.source == "speech"):
            contained = [
                ocr for ocr in ocr_segments
                if speech.start <= ocr.start and ocr.end <= speech.end
            ]
            if len(contained) < 2:
                continue

            self.metrics["long_speech_segments_detected"] += 1
            if any(
                TextSimilarityService.get_similarity(get_text(speech), get_text(ocr)) > 0.6
                for ocr in contained
            ):
                setattr(speech, "_long_speech_parent_id", speech.id)
                continue

            occupied = sorted(
                (
                    max(speech.start, ocr.start),
                    min(speech.end, ocr.end),
                )
                for ocr in contained
            )
            limit = min(speech.end, video_duration) if video_duration is not None else speech.end
            gaps = []
            cursor = speech.start
            for start, end in occupied:
                if start > cursor:
                    gaps.append((cursor, start))
                cursor = max(cursor, end)
            if limit > cursor:
                gaps.append((cursor, limit))

            eligible = [
                gap for gap in gaps
                if gap[1] - gap[0] >= self.config.min_duration
            ]
            if not eligible:
                continue

            active_chunks = self._partition_text(get_text(speech), len(eligible))
            source_chunks = self._partition_text(speech.text, len(active_chunks))
            source_chunks += [""] * (len(active_chunks) - len(source_chunks))
            translated_chunks = self._partition_text(speech.translated_text, len(active_chunks))
            translated_chunks += [""] * (len(active_chunks) - len(translated_chunks))
            split_segments = []
            for index, ((start, end), active_text) in enumerate(
                zip(eligible, active_chunks)
            ):
                split = Segment(
                    id=f"{speech.id}_long_{index}",
                    start=start,
                    end=min(end, start + self.config.max_duration),
                    text=source_chunks[index],
                    translated_text=translated_chunks[index],
                    source="speech",
                    confidence=speech.confidence,
                    confidence_available=speech.confidence_available,
                    bbox=speech.bbox,
                    metadata=dict(speech.metadata),
                )
                if is_translated:
                    split.translated_text = active_text
                else:
                    split.text = active_text
                setattr(split, "_long_speech_parent_id", speech.id)
                split_segments.append(split)
            replacements[speech.id] = split_segments
            self.metrics["long_speech_segments_split_or_queued"] += 1

        if replacements:
            segments[:] = [
                replacement
                for segment in segments
                for replacement in replacements.get(segment.id, [segment])
            ]

    @staticmethod
    def _partition_text(text: str, part_count: int) -> List[str]:
        words = text.split()
        if not words:
            return []
        if len(words) == 1 and part_count > 1:
            characters = list(words[0])
            count = min(part_count, len(characters))
            chunks = [
                "".join(
                    characters[index * len(characters) // count:
                               (index + 1) * len(characters) // count]
                )
                for index in range(count)
            ]
            return chunks
        count = min(part_count, len(words))
        chunks = [
            " ".join(words[index * len(words) // count:(index + 1) * len(words) // count])
            for index in range(count)
        ]
        return chunks

    def _resolve_overlaps(self, segments: List[Segment], get_text, replacement_required: bool = False) -> List[Segment]:
        if not segments:
            return []

        resolved = []
        for s in sorted(segments, key=lambda x: x.start):
            if not resolved:
                resolved.append(s)
                continue

            prev = resolved[-1]
            if s.start < prev.end:
                if replacement_required and prev.source != s.source:
                    resolved.append(s)
                    continue
                if (
                    replacement_required
                    and prev.source == "ocr"
                    and s.source == "ocr"
                    and not self._bboxes_intersect(prev.bbox, s.bbox)
                ):
                    # Spatially disjoint OCR tracks may overlap in time:
                    # they cover different screen regions simultaneously.
                    resolved.append(s)
                    continue
                # If nested overlap, we must resolve by score (cannot split outer)
                if s.end <= prev.end:
                    if self._caption_score(s, get_text) > self._caption_score(prev, get_text):
                        resolved.pop()
                        resolved.append(s)
                    else:
                        continue
                else:
                    # Partial overlap
                    if prev.source != s.source:
                        similarity = TextSimilarityService.get_similarity(get_text(prev), get_text(s))
                        if similarity > 0.6:
                            if s.source == "ocr":
                                continue
                            else:
                                resolved.pop()
                                resolved.append(s)
                                continue
                        else:
                            midpoint = (s.start + prev.end) / 2.0
                            if (midpoint - prev.start >= self.config.min_duration) and (s.end - midpoint >= self.config.min_duration):
                                prev.end = midpoint
                                s.start = midpoint
                                resolved.append(s)
                            elif midpoint - prev.start < self.config.min_duration:
                                new_end = prev.start + self.config.min_duration
                                if s.end - new_end >= self.config.min_duration:
                                    prev.end = new_end
                                    s.start = new_end
                                    resolved.append(s)
                                else:
                                    if s.source == "speech":
                                        resolved.pop()
                                        resolved.append(s)
                            else:
                                new_start = s.end - self.config.min_duration
                                if new_start - prev.start >= self.config.min_duration:
                                    prev.end = new_start
                                    s.start = new_start
                                    resolved.append(s)
                                else:
                                    if prev.source == "speech":
                                        pass
                                    else:
                                        resolved.pop()
                                        resolved.append(s)
                    else:
                        if self._caption_score(s, get_text) > self._caption_score(prev, get_text):
                            resolved.pop()
                            resolved.append(s)
                        else:
                            continue
            else:
                resolved.append(s)

        return resolved

    @staticmethod
    def _bboxes_intersect(a, b) -> bool:
        if not a or not b:
            # Unknown geometry: assume conflict (preserve old behavior).
            return True
        return not (
            a["right"] <= b["left"]
            or b["right"] <= a["left"]
            or a["bottom"] <= b["top"]
            or b["bottom"] <= a["top"]
        )

    @staticmethod
    def _caption_score(segment: Segment, get_text) -> tuple[float, float, int]:
        text = get_text(segment).replace("\n", " ").strip()
        clean_chars = sum(char.isalnum() or char.isspace() for char in text)
        cleanliness = clean_chars / max(len(text), 1)
        return (segment.confidence, cleanliness, len(text))