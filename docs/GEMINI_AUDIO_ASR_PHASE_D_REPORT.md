# Gemini Audio ASR Phase D Qualification Report

- **Status**: Phase D Completed (Qualification/Shadow-only Mode)
- **Production Ready**: NO
- **Input Method**: INLINE AUDIO (Base64 payload inline data in REST requests)
- **Inline Payload Safety Limit**: 19,000,000 bytes (~19 MB)
- **Remote Files API**: NOT IMPLEMENTED / DEFERRED (inline-only ASR fallback strategy)
- **Duration Parser**: Python standard library wave module
- **Full Audio Maximum Limit**: 300 seconds (5 minutes)
- **Chunk Fallback**: Valid WAV chunks partition, absolute timestamp conversion, overlap deduplication, and cleanup of temporary chunk files on success and exception.

---

## 1. Overview & Architecture

Phase D introduces GeminiAudioSpeechProvider as an alternative speech-to-text (ASR) engine implementing the standard SpeechProvider interface. In accordance with the project's Clean Architecture standards, this provider resides entirely within the Infrastructure layer (src/infrastructure/providers/speech/) and is registered in ProviderRegistry under "gemini_audio".

To prevent unintended production disruptions, the provider defaults to running in shadow-only/qualification mode. It does not replace the default production ASR provider (
aster-whisper or mock).

## 2. Key Technical Implementation Details

### Input Method & Payload Management
- **Base64 Inline Audio**: Audio payloads are sent inline via the REST request payload using base64 encoding rather than staging on Google's remote servers.
- **Safety Limit**: A safety ceiling of 19,000,000 bytes is enforced on the serialized JSON body. This avoids payload rejections or out-of-memory states on the client.
- **Remote Files API**: Use of Google Remote Files API is not implemented/deferred. The system relies entirely on inline data requests and fallback chunking.

### Audio Inspection & Validation
- **Wave Duration Parsing**: Audio duration, frame count, sample rate, and compression constraints are validated using the standard Python wave module.
- **Max Limit**: The maximum limit for a single inline audio file transcription is 300 seconds. Files longer than 300s automatically trigger chunk-based transcription fallback.

### Chunk Fallback Strategy
When a WAV file exceeds 300 seconds or the inline base64 limit, it is split into chunks of 60.0 seconds with an overlap of 2.0 seconds:
- **Temporary WAV Generation**: Valid WAV chunks are written using wave parameters identical to the source file.
- **Absolute Timestamps**: Timestamps returned by Gemini for each chunk (0-indexed offset) are mapped back to their absolute positions.
- **Overlap Deduplication**: Overlapping segments are resolved using text similarity metrics (SequenceMatcher ratio >= 0.85) to prevent duplicates.
- **Cleanup**: Temporary chunk files are cleaned up from the disk under both successful completion and exception/failure flows.

## 3. Known Limitations & Trade-offs
- **Free Tier Constraints**: Shared request scheduler caps rate limits globally across ASR, OCR, and Translation.
- **Memory Consumption**: Base64 encoding increases inline payload size by ~33%, but remains under the 19MB REST payload safety limit.
- **Strict Verbatim Prompt**: The prompt forces verbatim CJK Unicode outputs (e.g. 英国) and blocks correction of homophones or automatic grammar adjustments.
- **No Remote Files API**: Remote storage upload features are disabled, preserving data privacy on the Free Tier.

## 4. Phase E Handoff & Next Steps
- **Next Stage (Phase E)**: Comparative benchmark of
aster-whisper (small/base) vs gemini-3.1-flash-lite on accuracy, latency, and timestamp drift.
- **Qualification Script**: A new standalone qualification script scripts/qualify_gemini_asr.py was introduced to run shadow qualification runs.
