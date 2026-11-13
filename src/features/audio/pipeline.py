"""
Independent Audio Pipeline

Executes audio extraction, ASR transcription, shared Translation, TTS synthesis, and AudioComposer.
Independent from VisualPipeline: zero visual/YOLO/OCR imports.
"""

from __future__ import annotations

import json
import logging
import re
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from src.core.models.speech_segment import SpeechSegment
from src.features.audio.composer import AudioComposer
from src.features.audio.models import AudioRequest, AudioResult, TTSClip
from src.features.audio.ports import AudioExtractionPort, ASRCorrectionPort, ASRPort, TTSPort
from src.core.contracts import SpokenRewritePort
from src.features.translation.service import TranslationService
from src.features.translation.text_quality import (
    accept_spoken_line,
    is_already_target_language,
    is_collapsed_speech,
    repair_asr_phrases,
    restore_introduced_names,
    should_skip_spoken_translation,
)

logger = logging.getLogger(__name__)

_SENTENCE_BREAK = re.compile(r"(?<=[.!?…])\s+")


def split_packed_speech(
    segments: list[SpeechSegment],
    *,
    min_duration: float = 6.0,
) -> list[SpeechSegment]:
    """Turn one long multi-sentence window into one caption per sentence.

    A repaired 16-second ASR blob otherwise stays on screen as a single
    panel and covers the picture. Each sentence keeps a slice of that window.
    """
    output: list[SpeechSegment] = []
    for segment in segments:
        original = (segment.source_text or "").strip()
        previous = output[-1].source_text if output else ""
        text = _without_repeated_lead(previous, original)
        if original and not text:
            continue
        duration = segment.end_time - segment.start_time
        parts = [part.strip() for part in _SENTENCE_BREAK.split(text) if part.strip()]
        if len(parts) < 2 or duration < min_duration:
            if text and text != (segment.source_text or "").strip():
                segment.source_text = text
            output.append(segment)
            continue
        weights = [max(1, len(part)) for part in parts]
        total = float(sum(weights))
        consumed = 0.0
        starts = [segment.start_time]
        for weight in weights[:-1]:
            consumed += weight
            starts.append(segment.start_time + duration * (consumed / total))
        for index, part in enumerate(parts):
            start = starts[index]
            end = segment.end_time if index == len(parts) - 1 else starts[index + 1]
            if start >= segment.end_time - 0.05:
                output[-1].source_text = f"{output[-1].source_text} {part}".strip()
                continue
            output.append(
                SpeechSegment(
                    segment_id=f"{segment.segment_id}_{index}",
                    start_time=start,
                    end_time=max(start + 0.05, end),
                    source_text=part,
                    source_language=segment.source_language,
                    speaker_id=segment.speaker_id,
                )
            )
    return output


def _without_repeated_lead(previous: str, text: str) -> str:
    current = (text or "").strip()
    prior = (previous or "").strip()
    if not current or not prior:
        return current
    words = prior.split()
    for start in range(len(words)):
        suffix = " ".join(words[start:])
        if len(words) - start >= 2 and current.casefold().startswith(suffix.casefold()):
            return current[len(suffix):].lstrip(" ,.;")
    return current


class AudioPipeline:
    def __init__(
        self,
        asr_provider: Optional[ASRPort] = None,
        asr_corrector: Optional[ASRCorrectionPort] = None,
        translation_service: Optional[TranslationService] = None,
        spoken_rewriter: Optional[SpokenRewritePort] = None,
        tts_provider: Optional[TTSPort] = None,
        composer: Optional[AudioComposer] = None,
        audio_extractor: Optional[AudioExtractionPort] = None,
        max_rewrite_attempts: int = 3,
    ):
        self.asr_provider = asr_provider
        self.asr_corrector = asr_corrector
        self.translation_service = translation_service
        self.spoken_rewriter = spoken_rewriter
        self.tts_provider = tts_provider
        self.composer = composer or AudioComposer()
        self.audio_extractor = audio_extractor
        self.max_rewrite_attempts = max(0, int(max_rewrite_attempts))

    def process(self, request: AudioRequest, job_workspace_dir: str) -> AudioResult:
        start_time = time.monotonic()
        timings = {}
        warnings = []

        if not request.enabled:
            logger.info("AudioPipeline is disabled in request. Skipping execution.")
            return AudioResult(
                enabled=False,
                status="SKIPPED",
                warnings=["Audio processing disabled in request."],
            )

        ws_dir = Path(job_workspace_dir)
        audio_dir = ws_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        extracted_audio_path = str(audio_dir / "audio.wav")

        # 1. Audio Extraction
        t0 = time.monotonic()
        input_path = request.audio_path
        if input_path.lower().endswith((".mp4", ".mkv", ".avi", ".mov", ".webm")):
            if self.audio_extractor is None:
                raise RuntimeError(
                    "Audio extraction provider is not configured for video input."
                )
            extracted_audio_path = self.audio_extractor.extract_audio(
                input_path, extracted_audio_path
            )
        else:
            extracted_audio_path = input_path
        timings["audio_extraction"] = round(time.monotonic() - t0, 3)
        logger.info("Audio: audio track extracted (%.1fs)", timings["audio_extraction"])

        # 2. ASR Transcription (nếu chưa có segments từ người dùng)
        if request.segments is not None:
            speech_segments = list(request.segments)
            timings["asr"] = 0.0
        else:
            if self.asr_provider is None:
                raise RuntimeError("ASR provider is not configured for enabled AudioPipeline.")

            t0 = time.monotonic()
            logger.info("Audio: running ASR transcription (language=%s)...", request.source_language)
            speech_segments = self.asr_provider.transcribe(
                audio_path=extracted_audio_path, language=request.source_language
            )
            timings["asr"] = round(time.monotonic() - t0, 3)
            logger.info(
                "Audio: ASR done — %d speech segments (%.1fs)",
                len(speech_segments), timings["asr"],
            )

            # 2b. Contextual ASR correction (fail-open). A collapsed window may
            # be replaced by the secondary transcript's own timed lines.
            if self.asr_corrector is not None and speech_segments:
                t0 = time.monotonic()
                raw_snapshot = [segment.source_text for segment in speech_segments]
                try:
                    speech_segments = self.asr_corrector.correct_segments(
                        speech_segments,
                        language=request.source_language,
                        audio_path=extracted_audio_path,
                    )
                except Exception as exc:
                    warnings.append(f"ASR correction failed open: {exc}")
                    for segment, original in zip(speech_segments, raw_snapshot):
                        segment.source_text = original
                timings["asr_correction"] = round(time.monotonic() - t0, 3)
                logger.info("Audio: ASR contextual correction applied (%.1fs)", timings["asr_correction"])
            speech_segments = split_packed_speech(speech_segments)

        repaired_sources = restore_introduced_names(
            [repair_asr_phrases(segment.source_text or "") for segment in speech_segments]
        )
        for segment, source in zip(speech_segments, repaired_sources):
            segment.source_text = source
            if segment.translated_text:
                segment.translated_text = repair_asr_phrases(segment.translated_text)
            if segment.spoken_text:
                segment.spoken_text = repair_asr_phrases(segment.spoken_text)

        # 3. Shared Translation (chỉ dịch các segment chưa có bản dịch của người dùng)
        if self.translation_service is not None and speech_segments:
            t0 = time.monotonic()
            needed_indices = []
            skipped_as_target = 0
            for idx, seg in enumerate(speech_segments):
                if (seg.spoken_text or seg.translated_text).strip():
                    continue
                source = (seg.source_text or "").strip()
                if is_collapsed_speech(source):
                    warnings.append(
                        f"Segment {seg.segment_id} dropped: ASR text is missing its vowels."
                    )
                    seg.translated_text = ""
                    seg.spoken_text = ""
                    continue
                if should_skip_spoken_translation(
                    source,
                    request.target_language,
                    detected_language=seg.source_language,
                    job_source_language=request.source_language,
                ):
                    seg.translated_text = source
                    seg.spoken_text = source
                    skipped_as_target += 1
                    continue
                if source:
                    needed_indices.append(idx)
            if skipped_as_target:
                logger.info(
                    "Audio: %d spoken segment(s) already in '%s'; translator not called for them",
                    skipped_as_target,
                    request.target_language,
                )
            if needed_indices:
                raw_texts = [speech_segments[idx].source_text for idx in needed_indices]
                try:
                    translated_texts = self.translation_service.translate_texts(
                        raw_texts,
                        target_language=request.target_language,
                        source_language=request.source_language,
                    )
                except Exception as exc:
                    warnings.append(
                        "Spoken translation failed; Vietnamese lines keep their wording. "
                        + str(exc)[:240]
                    )
                    logger.warning("Audio: spoken translation failed open: %s", exc)
                    translated_texts = [""] * len(raw_texts)
                for idx, trans in zip(needed_indices, translated_texts):
                    accepted = accept_spoken_line(
                        speech_segments[idx].source_text,
                        trans,
                        request.target_language,
                    )
                    speech_segments[idx].translated_text = accepted
                    speech_segments[idx].spoken_text = accepted
                    if (
                        not accepted
                        and not request.dubbing_enabled
                        and (speech_segments[idx].source_text or "").strip()
                    ):
                        warnings.append(
                            f"Segment {speech_segments[idx].segment_id} left untranslated: "
                            "translation did not produce target-language speech."
                        )
            else:
                for seg in speech_segments:
                    if not seg.spoken_text and seg.translated_text:
                        accepted = accept_spoken_line(
                            seg.source_text, seg.translated_text, request.target_language
                        )
                        seg.spoken_text = accepted
            timings["translation"] = round(time.monotonic() - t0, 3)
            logger.info(
                "Audio: translated %d segments to '%s' (%.1fs)",
                len(needed_indices), request.target_language, timings["translation"],
            )
        elif speech_segments:
            for seg in speech_segments:
                if not seg.spoken_text and seg.translated_text:
                    seg.spoken_text = accept_spoken_line(
                        seg.source_text, seg.translated_text, request.target_language
                    )
                elif not seg.spoken_text and is_already_target_language(
                    seg.source_text, request.target_language
                ):
                    seg.spoken_text = seg.source_text.strip()

        # 4. Dubbing / TTS
        dubbed_audio_path = None
        tts_clips = []
        status = "SUCCESS"

        if request.dubbing_enabled:
            if self.tts_provider is None:
                warnings.append("Dubbing requested but TTS provider is not available. Reporting PARTIAL status.")
                status = "PARTIAL"
            else:
                t0 = time.monotonic()
                tts_dir = audio_dir / "tts"
                tts_dir.mkdir(parents=True, exist_ok=True)
                self._merge_short_spoken_tails(speech_segments)
                tts_total = sum(
                    1 for seg in speech_segments
                    if ((seg.spoken_text or seg.translated_text or seg.source_text) or "").strip()
                )
                logger.info("Audio: synthesizing %d TTS segments...", tts_total)
                # Each segment is its own HTTP call. Free TTS tiers reject more
                # than two in-flight requests (HTTP 429) and leave lines
                # undubbed. Results stay in segment order.
                workers = 1 if tts_total <= 1 else min(2, tts_total)
                missed: list = []
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = [
                        pool.submit(self._synthesize_segment, seg, request, tts_dir)
                        for seg in speech_segments
                    ]
                    for index, (seg, future) in enumerate(zip(speech_segments, futures), start=1):
                        clip, segment_warnings, failed = future.result()
                        synthesis_failed = failed and any(
                            item.startswith("TTS synthesis failed") for item in segment_warnings
                        )
                        if synthesis_failed:
                            missed.append(seg)
                        else:
                            warnings.extend(segment_warnings)
                            if failed:
                                status = "PARTIAL"
                            if clip is not None:
                                tts_clips.append(clip)
                        if index % 10 == 0 or index == len(futures):
                            tts_pct = int(100 * index / len(futures)) if futures else 100
                            logger.info(
                                "Audio: TTS %d/%d segments (%d%%)",
                                index, len(futures), tts_pct,
                                extra={"mediaflow_pct": tts_pct, "mediaflow_stage": "audio"},
                            )
                if missed:
                    logger.info("Audio: retrying %d TTS line(s) after the batch", len(missed))
                    time.sleep(1.0)
                    for seg in missed:
                        clip, segment_warnings, failed = self._synthesize_segment(
                            seg, request, tts_dir
                        )
                        if clip is not None:
                            tts_clips.append(clip)
                            continue
                        warnings.extend(segment_warnings)
                        status = "PARTIAL"

                timings["tts"] = round(time.monotonic() - t0, 3)
                logger.info("Audio: TTS done — %d clips in %.1fs", len(tts_clips), timings["tts"])

                if tts_clips:
                    t0 = time.monotonic()
                    dubbed_audio_path = str(audio_dir / "dubbed_timeline.wav")
                    source_duration = self._audio_duration(extracted_audio_path)
                    last_speech_end = speech_segments[-1].end_time if speech_segments else 0.0
                    mix_volume = float(request.mix_original_volume) if request.preserve_ambience else 0.0
                    dubbed_audio_path = self.composer.compose_dubbed_timeline(
                        tts_clips=tts_clips,
                        total_duration=max(source_duration, last_speech_end),
                        output_audio_path=dubbed_audio_path,
                        original_audio_path=extracted_audio_path,
                        mix_original_volume=mix_volume,
                    )
                    timings["audio_composition"] = round(time.monotonic() - t0, 3)
                    logger.info(
                        "Audio: dubbed timeline composed (%.1fs) in %.1fs",
                        max(source_duration, last_speech_end), timings["audio_composition"],
                    )
                elif speech_segments:
                    status = "PARTIAL"
                    warnings.append("Dubbing produced no usable TTS clips.")

        # Persist a non-secret, job-local audit after any duration-aware rewrites.
        clip_by_id = {clip.segment_id: clip for clip in tts_clips}
        try:
            (audio_dir / "segments.json").write_text(
                json.dumps(
                    [
                        {
                            "segment_id": seg.segment_id,
                            "start_time": seg.start_time,
                            "end_time": seg.end_time,
                            "source_text": seg.source_text,
                            "source_language": seg.source_language,
                            "translated_text": seg.translated_text,
                            "spoken_text": seg.spoken_text,
                            "tts_provider": clip_by_id.get(seg.segment_id).provider_name if seg.segment_id in clip_by_id else "",
                            "natural_tts_duration": round(clip_by_id.get(seg.segment_id).natural_duration, 4) if seg.segment_id in clip_by_id else 0.0,
                            "target_duration": round(clip_by_id.get(seg.segment_id).duration, 4) if seg.segment_id in clip_by_id else round(seg.end_time - seg.start_time, 4),
                            "speed_ratio": round(clip_by_id.get(seg.segment_id).speed_ratio, 4) if seg.segment_id in clip_by_id else 0.0,
                            "rewrite_count": clip_by_id.get(seg.segment_id).rewrite_count if seg.segment_id in clip_by_id else 0,
                        }
                        for seg in speech_segments
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            warnings.append(f"Could not write audio segments audit: {exc}")

        timings["total"] = round(time.monotonic() - start_time, 3)

        return AudioResult(
            enabled=True,
            status=status,
            extracted_audio_path=extracted_audio_path,
            dubbed_audio_path=dubbed_audio_path,
            segments=speech_segments,
            tts_clips=tts_clips,
            warnings=warnings,
            timings=timings,
        )

    @staticmethod
    def _merge_short_spoken_tails(segments, min_duration: float = 0.5) -> None:
        """Fold a sub-second scrap onto the previous line so TTS can speak it.

        A 0.4s tail such as "tranh nhau." cannot fit a natural clip under the
        speed cap. Joining it to the preceding utterance keeps the wording.
        """
        index = len(segments) - 1
        while index > 0:
            current = segments[index]
            previous = segments[index - 1]
            duration = current.end_time - current.start_time
            gap = current.start_time - previous.end_time
            extra = (
                current.spoken_text or current.translated_text or current.source_text or ""
            ).strip()
            if duration >= min_duration or gap > 0.2 or not extra:
                index -= 1
                continue
            base = (
                previous.spoken_text or previous.translated_text or previous.source_text or ""
            ).strip()
            joined = f"{base} {extra}".strip()
            previous.spoken_text = joined
            previous.translated_text = joined
            previous.source_text = (
                f"{(previous.source_text or '').strip()} {(current.source_text or '').strip()}"
            ).strip()
            previous.end_time = current.end_time
            segments.pop(index)
            index -= 1

    def _synthesize_segment(self, seg, request, tts_dir: Path):
        """Synthesize one line. Safe to run beside other lines: each clip has its own files."""
        text_to_speak = accept_spoken_line(
            seg.source_text,
            seg.spoken_text or seg.translated_text,
            request.target_language,
            detected_language=seg.source_language,
            job_source_language=request.source_language,
        )
        if not text_to_speak.strip():
            if (seg.source_text or seg.translated_text or "").strip():
                return None, [
                    f"Segment {seg.segment_id} left undubbed: no target-language line."
                ], True
            return None, [], False

        clip_path = str(tts_dir / f"tts_{seg.segment_id}.mp3")
        trimmed_path = tts_dir / f"trimmed_{seg.segment_id}.wav"
        target_duration = max(0.08, seg.end_time - seg.start_time)
        rewrite_count = 0
        seg_voice = request.voice
        if seg.speaker_id and request.speaker_voices and seg.speaker_id in request.speaker_voices:
            seg_voice = request.speaker_voices[seg.speaker_id]

        try:
            while True:
                raw_clip = self.tts_provider.synthesize(
                    text=text_to_speak, output_path=clip_path, voice=seg_voice
                )
                out_clip = str(self.composer.trim_tts_silence(Path(raw_clip), trimmed_path))
                natural_duration = self.composer.probe_duration(Path(out_clip))
                speed_ratio = natural_duration / target_duration if target_duration else 1.0
                if speed_ratio <= self.composer.max_speedup + 1e-6:
                    break
                if self.spoken_rewriter is None or rewrite_count >= self.max_rewrite_attempts:
                    raise RuntimeError(
                        f"natural TTS needs {speed_ratio:.2f}x speed-up; "
                        f"cap is {self.composer.max_speedup:.2f}x"
                    )
                rewritten = self.spoken_rewriter.rewrite_for_duration(
                    text=text_to_speak,
                    source_text=seg.source_text,
                    target_duration=target_duration,
                    current_duration=natural_duration,
                    target_language=request.target_language,
                ).strip()
                if not rewritten or rewritten == text_to_speak:
                    raise RuntimeError(
                        f"duration rewrite could not shorten a {speed_ratio:.2f}x TTS clip"
                    )
                text_to_speak = rewritten
                seg.spoken_text = rewritten
                rewrite_count += 1

            raw_provider = getattr(self.tts_provider, "last_provider_name", None)
            provider_name = (
                str(raw_provider) if isinstance(raw_provider, str) else type(self.tts_provider).__name__
            )
            return TTSClip(
                segment_id=seg.segment_id,
                start_time=seg.start_time,
                end_time=seg.end_time,
                audio_clip_path=out_clip,
                duration=target_duration,
                spoken_text=text_to_speak,
                natural_duration=natural_duration,
                speed_ratio=speed_ratio,
                rewrite_count=rewrite_count,
                provider_name=provider_name,
            ), [], False
        except Exception as exc:
            return None, [f"TTS synthesis failed for segment {seg.segment_id}: {exc}"], True

    @staticmethod
    def _audio_duration(path: str) -> float:
        """Return duration for the canonical extracted PCM WAV without FFmpeg coupling."""
        try:
            with wave.open(path, "rb") as handle:
                rate = handle.getframerate()
                return handle.getnframes() / float(rate) if rate else 0.0
        except (wave.Error, OSError):
            return 0.0
