# Provider Architecture

Providers are infrastructure adapters for external models, APIs, SDKs and system tools. Business features depend on provider contracts, not concrete provider implementations.

## 1. Provider boundaries

### DetectorPort

Purpose: detect the four burn-in visual classes on video frames.

Current production model baseline: YOLO26n fine-tune at `models/yolo/best.pt`.

The detector returns canonical `Detection` data. It does not decide blur, OCR, translation, or replacement.

### ASRPort

Conceptual method:

```text
transcribe(audio_path, options) → SpeechSegment[]
```

Known code/history may contain Faster-Whisper and Gemini Audio implementations. Provider selection must remain configurable; feature code must not depend on either implementation directly.

### OCRPort

Conceptual method:

```text
read_subtitle(crops / subtitle event evidence, options) → source text result
```

For V2 subtitle processing, YOLO already supplies the visual region. OCR should read supplied crops/evidence rather than rediscovering the global subtitle bbox as a second detector.

Subtitle OCR is local RapidOCR (ONNX PP-OCR) only. Gemini is not called to read crops. Translation stays on its own provider.

### TranslationPort

Conceptual method:

```text
translate(text units, target_language, options) → translated text units
```

Translation is shared by Audio and Subtitle.

It receives text only and cannot decide bbox, class, safety, masks, or rendering.

### TTSPort

Conceptual method:

```text
synthesize(translated_text, voice/options) → audio clip
```

TTS implementation(s) must be audited during the V2 migration. If production TTS is not currently complete, keep the port and report the capability as partial rather than mocking a successful production path.

### Video infrastructure

Video/media adapters may provide:

- metadata probing;
- frame decode/encode;
- audio extraction;
- audio/video muxing;
- codec selection/fallback;
- final invariant validation.

FFmpeg remains the preferred system boundary for encoding/muxing where practical.

## 2. Provider ownership rules

Providers must do only the work of their own capability.

Forbidden examples:

- OCR provider translating text.
- Translation provider reading images/audio.
- ASR provider using visual subtitle pixels.
- Detector deciding which class should be blurred.
- TTS provider deciding video timing or placement.
- Video renderer calling Gemini.

## 3. Lazy initialization

Providers must not be initialized when their feature is not required.

Examples:

```text
visual blur-only → detector + tracker + renderer; no Gemini/ASR/TTS
subtitle disabled → no OCR for subtitle
replace_image only → no OCR/Translation
Audio disabled → no ASR/TTS
Visual disabled → no YOLO
```

## 4. Provider selection

Keep selection simple.

A small composition function or mapping is sufficient:

```text
"faster-whisper" → FasterWhisperASRAdapter
"gemini-audio"   → GeminiAudioASRAdapter
"gemini"         → GeminiTranslationAdapter
```

A large auto-registration/plugin framework is unnecessary until there are enough providers to justify it.

## 5. Configuration and secrets

- Business feature config and provider config are separate concerns.
- API secrets are loaded from environment/secret storage, not serialized into job files.
- Model IDs, device, compute type and provider options are configuration values.
- Do not silently fallback to another provider on authentication/configuration failure unless the request explicitly enables a fallback policy.

## 6. Error contract

Adapters map low-level errors into a small MediaFlow provider error hierarchy while preserving causes.

Transient errors may be retried according to provider/runtime policy. Validation, authentication, unsupported-model and deterministic schema errors should fail explicitly.

Providers must never generate fake success data to keep the pipeline moving.

## 7. Testing providers

Each provider contract should have:

- a lightweight fake/mock used by unit tests;
- contract tests shared across implementations where practical;
- optional real-provider smoke tests guarded by environment/config;
- no required network/GPU for the default unit-test suite.
