# Feature Contracts

This document defines the boundaries between MediaFlow features. Names are conceptual contracts; exact Python class names may differ as long as the behavior and dependency ownership remain equivalent.

## 1. Core visual contracts

### BBox

Represents a pixel-space rectangle inside one video frame.

Required information:

```text
x1, y1, x2, y2
frame_width, frame_height
```

Rules:

- coordinates are clamped/validated against the frame;
- `x2 > x1`, `y2 > y1`;
- normalized representation may be included for serialization/debugging.

### Detection

Produced by the detector for one frame.

```text
class_id
class_name
confidence
bbox
frame_index
timestamp_seconds
```

Valid class mapping is defined in `YOLO.md`.

### Track

Associates same-class detections over time.

```text
track_id
class_id
class_name
start_time
end_time
mean_confidence
samples / bbox trajectory
```

Tracking must not mix different classes.

## 2. SubtitleEvent

A subtitle track can contain multiple text events because two different captions may appear consecutively at the same position.

`SubtitleEvent` therefore represents one semantic on-screen caption interval, not merely one spatial track.

Suggested fields:

```text
event_id
track_id
start_time
end_time
bbox_samples / anchor bbox
representative_frame_refs
source_text? 
translated_text?
```

### SubtitleEventSegmenter contract

Input: subtitle-class Track/detections.  
Output: one or more `SubtitleEvent` objects.

It splits events using deterministic temporal/visual evidence such as:

- disappearance/gap;
- meaningful bbox discontinuity;
- significant crop appearance/text change;
- optional temporal consensus.

It does not call Translation and does not render pixels.

## 3. SpeechSegment

ASR output unit.

Suggested fields:

```text
segment_id
start_time
end_time
source_text
source_language?
speaker_id?        # optional
translated_text?   # may be populated by shared translation
```

Rules:

- `end_time > start_time`;
- timestamps are monotonic or explicitly sortable;
- no fabricated text for silence/failed segments.

## 4. Translation contract

A shared Translation service accepts text-bearing units and returns translated text while preserving caller identity/timing metadata.

It must not:

- read images;
- read audio;
- create/change bbox values;
- decide whether a region is subtitle/logo/watermark;
- decide blur/replace actions.

Audio and Subtitle both depend on this contract.

## 5. AudioPipeline contract

### Input

```text
source video/audio path
source language (optional/configurable)
target language
enabled provider settings
```

### Output

```text
speech segments
translated speech segments when enabled
TTS clip metadata when enabled
composed audio path when TTS/dubbing completes
timings/warnings
```

### Pipeline

```text
AudioPipeline.process
  → extract audio if needed
  → ASRPort.transcribe
  → Translation service
  → TTSPort.synthesize
  → AudioComposer.compose
```

### Independence

AudioPipeline must be unit-testable without video frames, YOLO or VisualPipeline.

## 6. AudioComposer contract

Input:

- original duration/timeline metadata;
- speech segment start/end times;
- synthesized TTS clips.

Output:

- one audio stream aligned to the intended video timeline.

Responsibilities may include:

- silence insertion;
- clip placement;
- overlap policy;
- bounded time-stretch/trimming policy if explicitly supported;
- final duration normalization.

TTS providers only synthesize audio; they do not own timeline composition.

## 7. VisualPipeline contract

### Input

```text
source video
YOLO/detection config
per-class feature config
```

### Output

```text
detections
tracks
subtitle events when enabled
rendered visual stream/video
action/timing summaries
```

### Pipeline

```text
Detector
→ Tracker
→ route by class
→ processor
→ shared actions
→ renderer
```

VisualPipeline must be runnable with Audio disabled.

## 8. Visual class processors

### SubtitleProcessor

Input: subtitle tracks/events.  
Default behavior: `translate_replace`.

```text
SubtitleEvent
→ crop source evidence
→ OCRPort
→ Translation service
→ background action (default blur)
→ TextOverlayAction
```

Rendering uses source timing and source spatial anchor.

### LogoProcessor

Input: logo tracks.  
Default: `BlurAction`.

May select implemented alternatives such as `PreserveAction` or `ReplaceImageAction` through configuration.

### WatermarkProcessor

Input: watermark tracks.  
Default: `BlurAction`.

Does not invoke OCR/Translation by default.

### OtherTextProcessor

Input: `other_burn_in_text` tracks.  
Default: `BlurAction`.

It is not silently routed into SubtitleProcessor.

## 9. Shared visual actions

Each action performs one rendering responsibility and is independently testable.

### BlurAction

Input: frame + bbox/action parameters.  
Output: frame/ROI with real pixel blur.

### PreserveAction

No pixel change.

### SolidCoverAction

Optional retained action that fills a bounded region with a configured color.

### ReplaceImageAction

Composites an image asset into the target region with explicit fit/opacity rules.

### TextOverlayAction

Draws Unicode text using a supplied target anchor/box and style.

For subtitle replacement it must support:

- original subtitle anchor/box;
- adaptive wrapping;
- adaptive font size;
- frame-bound clipping;
- deterministic alignment;
- moving bbox trajectories when required.

## 10. ProcessingRequest

The public request should remain product-oriented.

Suggested shape:

```json
{
  "input_video": "input.mp4",
  "output_video": "output.mp4",
  "job_id": null,
  "device": "auto",
  "audio": {
    "enabled": false,
    "asr": {"provider": "faster-whisper"},
    "translation": {"provider": "gemini", "target_language": "vi"},
    "tts": {"provider": null}
  },
  "visual": {
    "enabled": true,
    "subtitle": {
      "enabled": true,
      "mode": "translate_replace",
      "background": "blur",
      "target_language": "vi"
    },
    "logo": {"enabled": true, "action": "blur"},
    "watermark": {"enabled": true, "action": "blur"},
    "other_burn_in_text": {"enabled": true, "action": "blur"}
  },
  "overwrite": true
}
```

Provider-specific model/options may be configured separately so the frontend does not need to know internal Python class names.

## 11. ProcessingResult

Must be serializable and suitable for CLI/API use.

Recommended information:

```text
job_id
status
input/output paths
media invariants
detection/track/event summaries
audio summary
visual action summary
timings
warnings
error
workspace path
```

Do not overload one action counter to represent unrelated concepts. When useful, distinguish:

- tracks/events processed;
- operations rendered;
- ignored/disabled items.
