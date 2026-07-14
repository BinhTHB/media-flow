from typing import List

from src.domain.entities.segment import Segment


class SpeechQualityAuditor:
    LOW_CONFIDENCE_THRESHOLD = 0.5
    LOW_CHARS_PER_SECOND = 0.5

    @classmethod
    def report(
        cls,
        asr_segments: List[Segment],
        translation_input: List[Segment],
        translated_segments: List[Segment],
        language: str,
        detected_language: str | None = None,
    ) -> dict:
        speech_input = [segment for segment in translation_input if segment.source == "speech"]
        speech_output = [segment for segment in translated_segments if segment.source == "speech"]
        asr_text = "".join(segment.text.strip() for segment in asr_segments)
        translated_text = "".join(segment.translated_text.strip() for segment in speech_output)
        asr_duration = sum(segment.end - segment.start for segment in asr_segments)
        chars_per_second = len(asr_text) / asr_duration if asr_duration > 0 else 0.0
        low_confidence = [
            segment for segment in asr_segments
            if segment.confidence < cls.LOW_CONFIDENCE_THRESHOLD
        ]
        low_information = len(asr_segments) == 1 and len(asr_text) <= 4
        collapsed = (
            len(speech_output) < len(speech_input)
            or (bool(asr_segments) and not speech_output)
            or low_information
        )
        warnings = []
        if not asr_text:
            warnings.append("empty_or_near_empty_transcript")
        if low_information:
            warnings.append("low_information_asr_output")
        if low_confidence:
            warnings.append("low_confidence_asr_output")
        if any(segment.confidence == 0.0 for segment in asr_segments):
            warnings.append("zero_confidence_asr_output")
        if chars_per_second < cls.LOW_CHARS_PER_SECOND and asr_duration > 0:
            warnings.append("very_low_chars_per_second")
        if (
            len(asr_segments) == 1
            and asr_duration >= 10.0
            and len(asr_text) <= 10
        ):
            warnings.append("single_long_segment_with_tiny_text")
            warnings.append("duration_text_inconsistency")
        texts = [segment.text.strip() for segment in asr_segments if segment.text.strip()]
        if len(texts) != len(set(texts)):
            warnings.append("repeated_hallucinated_text")
        if (
            language != "auto"
            and detected_language
            and detected_language != language
        ):
            warnings.append("language_mismatch")
        if len(speech_output) < len(speech_input):
            warnings.append("speech_segment_count_collapsed")

        return {
            "asr_segment_count": len(asr_segments),
            "asr_total_duration": asr_duration,
            "asr_text_length": len(asr_text),
            "asr_chars_per_second": chars_per_second,
            "asr_language": language,
            "asr_detected_language": detected_language,
            "asr_low_confidence_count": len(low_confidence),
            "speech_translation_input_count": len(speech_input),
            "speech_translation_output_count": len(speech_output),
            "speech_text_length_before_translation": sum(
                len(segment.text.strip()) for segment in speech_input
            ),
            "speech_text_length_after_translation": len(translated_text),
            "speech_collapse_detected": collapsed,
            "warnings": warnings,
            "asr_segments": [segment.to_dict() for segment in asr_segments],
            "translated_speech_segments": [segment.to_dict() for segment in speech_output],
        }
