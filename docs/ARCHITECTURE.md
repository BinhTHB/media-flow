# MediaFlow Architecture V2

## 1. Architectural style

MediaFlow uses a **Feature-first Modular Clean Architecture**.

The project is one deployable application (a modular monolith), but its major capabilities are isolated so they can be enabled, disabled, tested, replaced, and evolved independently.

The architecture combines two ideas:

1. **Feature-first modules** — organize business code around Audio, Visual, Subtitle and Translation rather than around a long generic processing framework.
2. **Clean Architecture dependency rules** — business features depend on contracts; external libraries and APIs implement those contracts at the infrastructure boundary.

The goal is not maximum abstraction. The goal is maximum practical independence with a small, understandable call graph.

## 2. Canonical processing architecture

```text
                         Interface
                    CLI / local Web API
                              │
                              ▼
                         Application
                      ProcessVideo service
                              │
                 ┌────────────┴────────────┐
                 │                         │
                 ▼                         ▼
           AudioPipeline              VisualPipeline
                 │                         │
        ASR → Translation → TTS       YOLO → Tracking
                 │                         │
          AudioComposer          ┌─────────┼───────────┐
                 │               │         │           │
                 │           Subtitle     Logo    Watermark/Other
                 │               │         │           │
                 │          EventSegmenter └──── shared Actions
                 │               │
                 │              OCR
                 │               │
                 └──────────┐    │
                            ▼    ▼
                       Translation
                            │
                        Render/Mux
                            │
                        Final Video
```

## 3. Major modules

### 3.1 AudioPipeline

Audio is an independent feature.

```text
video/audio
→ extract audio when required
→ ASR
→ source speech segments
→ shared Translation service
→ translated speech segments
→ TTS
→ AudioComposer
→ dubbed audio timeline
```

AudioPipeline must not import or initialize YOLO, VisualPipeline, subtitle OCR, or visual actions.

If Audio is disabled, ASR and TTS must not be initialized.

### 3.2 VisualPipeline

Visual processing is independent from Audio.

```text
video frames
→ YOLO 4-class detection
→ class-aware tracking
→ class router
   ├── subtitle
   ├── watermark
   ├── logo
   └── other_burn_in_text
→ feature processors
→ rendering actions
→ rendered video stream
```

VisualPipeline must run with the original audio preserved even when AudioPipeline is disabled.

### 3.3 SubtitleProcessor

Subtitle is a special visual feature because it needs semantic text processing.

```text
YOLO subtitle detections
→ SubtitleTracker / tracked geometry
→ SubtitleEventSegmenter
→ SubtitleEvent(start, end, bbox trajectory)
→ select/crop representative source frames
→ OCR
→ source text
→ shared Translation service
→ translated text
→ BlurAction on original subtitle region
→ TextOverlayAction anchored to original subtitle position
```

Rules:

- YOLO owns visual detection geometry and class.
- Tracking owns temporal geometry continuity.
- `SubtitleEventSegmenter` owns splitting one spatial track into distinct text events when the text changes over time.
- OCR reads text from the supplied subtitle crops. OCR does not redefine the YOLO bbox or decide render position.
- Translation receives text only. It never reads images/audio, creates bbox values, or decides rendering.
- The renderer uses the source geometry. It may adapt font size, wrapping, and bounded width so Vietnamese text fits, but it must preserve the source anchor/region rather than moving every subtitle to a global bottom-center position.

### 3.4 LogoProcessor, WatermarkProcessor, OtherTextProcessor

These processors are intentionally small. They route their class tracks to reusable actions.

Default behavior:

```text
logo               → blur
watermark          → blur
other_burn_in_text → blur
```

Supported policy may later select another implemented action such as `preserve`, `replace_image`, or `solid_cover` without changing YOLO or the other processors.

Processors do not implement Gaussian blur, image compositing, or text drawing themselves.

### 3.5 Shared actions

Reusable pixel/render actions live once and are shared by visual processors:

- `BlurAction`
- `PreserveAction`
- `SolidCoverAction` when retained
- `ReplaceImageAction`
- `TextOverlayAction`

Do not duplicate blur/image/text algorithms inside class-specific processors.

A simple action lookup/registry is acceptable. A generic plugin framework, arbitrary DAG, or workflow engine is not required for V2.

### 3.6 Shared Translation

Translation is a shared capability used by both Audio and Visual Subtitle processing.

```text
AudioPipeline ─────────┐
                       ├── TranslationPort → provider adapter
SubtitleProcessor ─────┘
```

There must not be separate business translation implementations for Audio and Subtitle.

## 4. Clean Architecture boundaries

Recommended target organization:

```text
src/
├── core/
│   ├── models/
│   ├── errors.py
│   └── contracts.py
├── features/
│   ├── audio/
│   │   ├── pipeline.py
│   │   ├── ports.py
│   │   ├── models.py
│   │   └── composer.py
│   ├── visual/
│   │   ├── pipeline.py
│   │   ├── ports.py
│   │   ├── tracker.py
│   │   ├── subtitle/
│   │   │   ├── processor.py
│   │   │   ├── event_segmenter.py
│   │   │   └── renderer.py
│   │   ├── processors/
│   │   │   ├── logo.py
│   │   │   ├── watermark.py
│   │   │   └── other_text.py
│   │   └── actions/
│   │       ├── blur.py
│   │       ├── preserve.py
│   │       ├── solid_cover.py
│   │       ├── replace_image.py
│   │       └── text_overlay.py
│   └── translation/
│       ├── service.py
│       └── port.py
├── application/
│   ├── process_video.py
│   ├── request.py
│   └── result.py
├── infrastructure/
│   ├── detection/
│   ├── asr/
│   ├── ocr/
│   ├── translation/
│   ├── tts/
│   ├── gemini/
│   └── video/
└── interfaces/
    ├── composition/
    ├── jobs/
    └── web/
```

Exact filenames may differ when reuse makes another layout cleaner. The dependency rules are more important than folder names.

## 5. Dependency rules

Allowed direction:

```text
core ← features ← application ← interfaces
  ↑        ↑
  └────────┴──── infrastructure implements feature/application ports
```

Mandatory rules:

- `core` imports no Ultralytics, OpenCV, Gemini, Faster-Whisper, TTS SDK, FFmpeg wrapper, or UI framework.
- `features/audio` must not import VisualPipeline or YOLO.
- `features/visual` must not import ASR/TTS implementations.
- `features/visual/subtitle` must not import Gemini SDK/client directly; it depends on OCR/Translation contracts.
- Infrastructure adapters may import third-party libraries.
- CLI/API only parse requests, build dependencies, invoke the application service, and serialize results.
- Production feature code must not import benchmark/evaluation namespaces.

## 6. Ports and adapters

Use a port when an external implementation is genuinely replaceable or optional:

- Detector
- ASR
- OCR
- Translation
- TTS
- video encode/mux as appropriate

Do not create an interface only because Clean Architecture exists. Internal algorithms with one stable implementation can remain ordinary classes/functions until replaceability is needed.

## 7. Application orchestration

`process_video(request)` is a thin composition boundary.

Conceptual flow:

```text
validate request
prepare job workspace
if visual.enabled:
    visual_result = VisualPipeline.process(...)
if audio.enabled:
    audio_result = AudioPipeline.process(...)
combine selected visual stream + selected audio stream
validate output invariants
return ProcessingResult
```

The application service must not contain per-class YOLO business rules, OCR prompts, TTS synthesis logic, or pixel algorithms.

## 8. Job workspace and artifacts

Keep job isolation because it is useful for CLI today and API/jobs later.

Canonical root:

```text
outputs/jobs/<job_id>/
```

Recommended artifacts:

```text
job.json
result.json
detections.json
tracks.json
subtitle_events.json        # only when subtitle processing runs
transcript.json             # only when audio ASR runs
translations.json           # only when translation runs
render_plan.json            # optional diagnostic artifact
logs/
```

An action/render plan is a diagnostic/intermediate representation, not the central business architecture.

Generated artifacts must be gitignored.

## 9. Configuration model

Prefer a product-shaped request rather than a generic workflow graph:

```json
{
  "audio": {
    "enabled": false,
    "asr": {"provider": "..."},
    "translation": {"target_language": "vi"},
    "tts": {"provider": "..."}
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
  }
}
```

Runtime/request override wins over environment, which wins over project config, which wins over code defaults. Do not maintain two conflicting business-policy configuration files.

## 10. Lazy execution

Disabled/unneeded capabilities must not initialize expensive dependencies.

Examples:

- Audio disabled → no ASR/TTS.
- Visual disabled → no YOLO.
- Subtitle disabled and other classes only blur → no OCR/Translation.
- `replace_image` → no OCR/ASR/Translation.
- Subtitle `translate_replace` → OCR + Translation, not ASR by default.

## 11. Intentional non-goals

V2 does not implement a generic workflow/DAG engine, third-party plugin platform, event bus, arbitrary node editor, or microservice split.

Those should be introduced only when a real product requirement requires user-defined arbitrary workflows, many independent teams, distributed execution, or external plugins.
