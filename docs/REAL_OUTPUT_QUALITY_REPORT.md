# Real Output Quality Report

## 1. Summary

The full `sample.mp4` pipeline completed with Faster-Whisper `tiny`, Gemini
3.1 Flash Lite, ASS subtitles, and FFmpeg rendering. The output video is
technically valid and Vietnamese subtitles are visibly burned in, but the
translation is not yet release quality because ASR mistakes propagate directly
into the Vietnamese text.

## 2. Artifact Checklist

Job directory: `output/job_demo_123`

| Artifact | Result |
| --- | --- |
| `audio/audio.wav` | Present, 8,464,122 bytes |
| `transcripts/transcript.json` | Present |
| `transcripts/translated.json` | Present |
| `transcripts/normalized.json` | Present |
| `transcripts/segmented.json` | Present |
| `subtitles/subtitle.srt` | Present |
| `subtitles/subtitle.ass` | Present |
| `videos/output.mp4` | Present, 70,893,425 bytes |
| `logs/job.log` | Missing; the CLI output is the available runtime record |

## 3. Transcript Metrics

| Metric | Result |
| --- | --- |
| Speech segments | 160 |
| Average duration | 1.576s |
| Shortest / longest | 0.640s / 3.120s |
| Empty text / adjacent duplicates | 0 / 0 |
| Timestamp overlaps | 0 |
| Gaps over 0.1s | 8; 12.080s total; 3.000s longest |
| Segments under 0.8s | 2 |

The timestamps are ordered and non-overlapping. The gaps are consistent with
dialogue pauses. All ASR confidence values are `0.0`, so they cannot support
automated confidence-based triage.

Quality risk: the `tiny` model produces garbled Chinese in several places. For
example, contextually likely `因果` (karma) is transcribed as `英国` (England),
which produces incorrect downstream Vietnamese.

## 4. Translation Metrics

| Metric | Result |
| --- | --- |
| Translated segments | 160 |
| Empty translations | 0 |
| Untranslated / source-identical translations | 1 |
| Vietnamese translations containing CJK text | 1 |

`speech_150` retains Chinese text (`原来清静和谨排还算换数`) in
`translated_text`; it propagates into downstream subtitle artifacts.

The primary quality limitation is not throughput or subtitle generation:
Gemini translates the ASR text it receives, including incorrect homophones and
garbled phrases. Sample frame text also shows Vietnamese that is grammatical
in places but semantically inconsistent with the scene.

## 5. Subtitle Metrics

| Metric | Result |
| --- | --- |
| Normalized segments | 52 |
| Final segmented subtitles | 90 |
| SRT entries / ASS dialogue events | 90 / 90 |
| Maximum characters per line | 42 |
| Maximum lines per subtitle | 2 |
| Segmented timestamp overlaps | 0 |
| Duplicate subtitle text | 0 |
| Normalized-to-segmented content check | Pass |

The initial validation run exposed short tail subtitles created by proportional
splitting. `SubtitleSegmentationService` now reserves the configured minimum
duration for remaining chunks. The rerun reduces the minimum segmented
duration to the configured 0.8s threshold (SRT millisecond rounding leaves two
entries at 0.799s).

## 6. ASS Style and Render Sanity

The ASS file uses `Arial` 28pt, white text, outline `3.0`, shadow `1.5`, and
two-line maximum formatting at 1920x1080. Three frames were extracted:

- `validation/frames/early.jpg`
- `validation/frames/middle.jpg`
- `validation/frames/late.jpg`

They confirm Vietnamese subtitles are rendered and readable. The source video
also contains embedded Chinese subtitles, so the Vietnamese subtitle occupies
the lower area beneath/near the source subtitle. This is legible but visually
busy; it is not an ASS rendering failure.

`output.mp4` contains one H.264 video stream and one AAC audio stream. Its
duration is 264.496s, matching the 264.5s audio duration within 0.004s.

## 7. Runtime Review

| Stage | Time |
| --- | --- |
| Audio extraction | 0.4s |
| Faster-Whisper `tiny` STT | 29.6s |
| Gemini translation | 31.1s |
| Subtitle generation | 0.0s |
| FFmpeg render | 89.4s |
| Total | ~150.5s |

FFmpeg rendering is the runtime bottleneck, but it is acceptable for a
4m24s 1080p video. Gemini rate limiting is working with 20-segment batches
and a 4.5s inter-request gap for the 15 RPM model quota. Do not change
batching unless quota behavior changes.

## 8. Top Quality Issues

1. **High:** ASR quality from Faster-Whisper `tiny` is insufficient for this
   Chinese dialogue; mistranscriptions produce materially wrong Vietnamese.
2. **Medium:** One translated segment remains Chinese and reaches SRT/ASS.
3. **Low:** All confidence values are zero, preventing confidence-based review.
4. **Low:** Source Chinese subtitles plus burned Vietnamese subtitles are
   visually dense.

## 9. Applied Small Fix

`SubtitleSegmentationService` now prevents proportional splitting from
starving a trailing chunk below the configured minimum subtitle duration when
the parent segment has enough time for all chunks.

## 10. Recommended Next Milestone

Validate the same clip with Faster-Whisper `base` or `small`, then compare the
ASR transcript and Vietnamese semantic accuracy against this baseline. Add a
small post-translation quality gate that flags leftover CJK text for review or
retry. No new provider, REST API, TTS, or dubbing is required for that
milestone.
