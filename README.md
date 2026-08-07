# AI Video Translator

Hệ thống dịch video short (tiếng Trung/Anh → Việt) theo kiến trúc **Clean Architecture + Feature-based modules**.

## Kiến trúc

```
src/
├── domain/                # Core business logic - không phụ thuộc gì
│   ├── entities/          # VideoJob, Segment, SubtitleItem, FeatureConfig
│   └── value_objects/
├── application/           # Use cases & ports (interfaces)
│   ├── ports/             # SpeechProvider, OCRProvider, TranslationProvider, etc.
│   ├── use_cases/         # ExtractAudio, TranscribeSpeech, DetectOnscreenText, ...
│   └── orchestrator/      # JobOrchestrator điều phối pipeline linh hoạt
├── infrastructure/        # Cài đặt cụ thể của providers (mock, real)
│   └── providers/
│       ├── speech/        # MockSpeechProvider, FasterWhisperSpeechProvider
│       ├── ocr/           # MockOCRProvider
│       ├── translation/   # MockTranslationProvider
│       ├── subtitle/      # MockSubtitleProvider
│       ├── video/         # MockVideoProvider
│       └── storage/       # MockStorageProvider
├── interfaces/            # Giao tiếp với bên ngoài (CLI, API)
│   ├── cli/demo.py
│   └── api/
├── tests/                 # Unit tests
│   ├── test_validation.py
│   ├── test_use_cases.py
│   └── test_orchestrator.py
└── shared/                # Exception classes
```

## Nguyên tắc thiết kế

- **Domain/application không import trực tiếp package hạ tầng** (Whisper, PaddleOCR, FFmpeg, OpenAI, Gemini...).
- Mọi công cụ bên ngoài đi qua **interface/port** trong `application/ports/`.
- **JobOrchestrator** kiểm tra feature nào bật rồi chạy đúng use case tương ứng.
- Pipeline không hard-code: mỗi use case nhận provider qua dependency injection.
- Validation ở domain layer: `Segment`, `FeatureConfig`, `VideoJob` có `__post_init__` kiểm tra dữ liệu đầu vào.

### Các provider có thể thay thế

| Interface | Mock | Real hiện tại |
|---|---|---|
| SpeechProvider | MockSpeechProvider | FasterWhisperSpeechProvider |
| OCRProvider | MockOCRProvider | - |
| TranslationProvider | MockTranslationProvider | - |
| VideoProvider | MockVideoProvider | - |
| SubtitleProvider | MockSubtitleProvider | - |
| StorageProvider | MockStorageProvider | - |

## Run Policy-Driven End-to-End Media Processing

Pipeline xử lý video theo kiến trúc **Class-Aware Policy-Driven Architecture**:
`Video Input` → `YOLO 4-class Detection` → `Class-Aware Spatial-Temporal Tracking` → `Action Policy Engine` → `Rendering Plan` → `Lazy Enrichment` → `Action Video Renderer` → `Final MP4`.

### Lệnh chạy E2E chính thức (Default: Blur-all 100% Offline)

```powershell
.venv\Scripts\python.exe scripts/run_e2e.py `
  --input test_videos
eal_mini_benchmark\sample5.mp4 `
  --output outputs\e2e_blur_sample5.mp4 `
  --preset blur-all `
  --device auto
```

### Lệnh dịch phụ đề sang tiếng Việt (Translate-subtitle preset)

```powershell
.venv\Scripts\python.exe scripts/run_e2e.py `
  --input test_videos
eal_mini_benchmark\sample5.mp4 `
  --output outputs\e2e_translate_sample5.mp4 `
  --preset translate-subtitle `
  --device auto
```

### Chạy với cấu hình policy tùy chỉnh (Custom Policy YAML)

```powershell
.venv\Scripts\python.exe scripts/run_e2e.py `
  --input test_videos
eal_mini_benchmark\sample5.mp4 `
  --output outputs\e2e_custom_sample5.mp4 `
  --policy config\custom_policy.yaml `
  --device auto
```

### Tham số E2E
- `--input`, `-i`: Đường dẫn video đầu vào thực tế (bắt buộc).
- `--output`, `-o`: Đường dẫn video đầu ra hoàn chỉnh (bắt buộc).
- `--preset`: Preset chính sách có sẵn (`blur-all` mặc định, `translate-subtitle`, `preserve-all`).
- `--policy`: Đường dẫn file cấu hình YAML policy tùy chỉnh.
- `--device`: Thiết bị tính toán `auto`, `cuda` hoặc `cpu` (mặc định: `auto`).

### Yêu cầu & Model
- **YOLO Weights**: `models/yolo/best.pt` (nạp 4 class: subtitle, watermark, logo, other_burn_in_text).
- **FFmpeg & FFprobe**: Đã cài đặt trong PATH hệ thống.
- **Biến môi trường**: `GEMINI_API_KEY` chỉ bắt buộc khi action của policy là `replace_text`. Chế độ `blur-all` mặc định chạy 100% offline mà không cần API Key.
- **Artifacts**: Mỗi job xuất kế hoạch hành động máy đọc được tại `outputs/action_plan.json`.

---

## Cách chạy demo

### Yêu cầu
- Python 3.10+

### Chạy với mock providers (không cần GPU, không gọi API thật)

```bash
python src/interfaces/cli/demo.py video.mp4 --stt --ocr --translate --subtitle
```

### Chạy với faster-whisper thật

```bash
# Cài đặt faster-whisper trước
pip install faster-whisper

# Chạy với STT thật (model base, chạy CPU)
python src/interfaces/cli/demo.py video.mp4 --stt --translate --subtitle \
    --speech-provider faster-whisper --whisper-model base
```

> **Lưu ý GitHub Actions CI**: Nên dùng model `tiny`, `base`, hoặc `small`. Không dùng `large` vì thời gian chạy quá lâu và tiêu tốn tài nguyên.

### Flags

| Flag | Chức năng |
|---|---|
| `--stt` | Speech-to-text (trích xuất lời nói) |
| `--ocr` | Nhận diện chữ trên màn hình |
| `--translate` | Dịch các segments sang target language |
| `--subtitle` | Tạo file SRT |
| `--render` | Render video với phụ đề (mock) |
| `--source-lang` | Ngôn ngữ nguồn (zh/en/auto, mặc định: zh) |
| `--target-lang` | Ngôn ngữ đích (mặc định: vi) |
| `--speech-provider` | Provider STT: `mock` (mặc định) hoặc `faster-whisper` |
| `--whisper-model` | Kích thước model: `tiny`, `base` (mặc định), `small`, `medium` |

### Ví dụ

```bash
# Chỉ STT + dịch (mock)
python src/interfaces/cli/demo.py video.mp4 --stt --translate

# STT + OCR + dịch + subtitle (mock)
python src/interfaces/cli/demo.py video.mp4 --stt --ocr --translate --subtitle

# Tất cả tính năng (mock)
python src/interfaces/cli/demo.py video.mp4 --stt --ocr --translate --subtitle --render

# STT thật với faster-whisper (model tiny, chạy CPU nhẹ)
python src/interfaces/cli/demo.py video.mp4 --stt --translate --subtitle \
    --speech-provider faster-whisper --whisper-model tiny
```

Output JSON hiển thị segments đã xử lý. Nếu bật `--subtitle`, file `output.srt` được tạo ra.

## Cách chạy unit tests

```bash
pytest
```

## Cách thay mock provider bằng provider thật

1. Tạo class implement interface tương ứng trong `src/infrastructure/providers/<loại>/`.
2. Inject vào use case thay vì mock provider.

Ví dụ đã có `FasterWhisperSpeechProvider`:

```python
from src.infrastructure.providers.speech.faster_whisper_speech_provider import FasterWhisperSpeechProvider

speech_provider = FasterWhisperSpeechProvider(model_size="base")
transcribe_uc = TranscribeSpeechUseCase(speech_provider)
```

Hoặc qua CLI:

```bash
python src/interfaces/cli/demo.py video.mp4 --stt --translate --subtitle \
    --speech-provider faster-whisper --whisper-model base
```

Nếu chưa cài `faster-whisper`, provider sẽ báo lỗi rõ ràng:

```
ProviderError: faster-whisper is not installed. Install it with: pip install faster-whisper
```

## Vì sao thiết kế này không hard-code pipeline?

- **FeatureConfig** cho phép bật/tắt từng tính năng độc lập.
- **JobOrchestrator** kiểm tra feature flag và chỉ chạy use case tương ứng.
- Mỗi **use case** nhận **provider interface** qua constructor (dependency injection).
- Pipeline có thể thay đổi mà không cần sửa domain/application code.
- Thêm provider mới không ảnh hưởng đến các provider khác.
- Dễ test: inject mock provider vào use case.
- Domain entities không phụ thuộc vào bất kỳ thư viện bên ngoài nào.

## Project Governance

Dự án này tuân thủ governance để giữ kiến trúc ổn định trong quá trình phát triển:

- [AGENTS.md](AGENTS.md): handbook cho AI coding agents và developer khi làm việc trong repository.
- [docs/PROJECT_OPERATING_SPECIFICATION.md](docs/PROJECT_OPERATING_SPECIFICATION.md): technical constitution định nghĩa nguyên tắc kiến trúc, feature system, provider specification và data contracts.

---

# Production MVP: Real Video → Subtitle

This milestone makes the project capable of processing a real short video (under 5 minutes) from audio extraction to subtitle generation.

## Installation

### Required Dependencies

```bash
# Python 3.10+
pip install -r requirements.txt  # (or install manually below)
```

Manual dependencies:
```bash
# Audio extraction
pip install faster-whisper

# Translation
pip install openai

# Config parsing
pip install pyyaml
```

### FFmpeg Installation

FFmpeg is required for audio extraction. Install it:

- **Windows**: `winget install Gyan.FFmpeg` or download from https://ffmpeg.org/download.html
- **macOS**: `brew install ffmpeg`
- **Linux (Debian/Ubuntu)**: `apt-get install ffmpeg`

Verify installation:
```bash
ffmpeg -version
```

### API Key

For OpenAI translation, set the `OPENAI_API_KEY` environment variable:

```bash
# Linux/macOS
export OPENAI_API_KEY=sk-...

# Windows PowerShell
$env:OPENAI_API_KEY="sk-..."

# Windows CMD
set OPENAI_API_KEY=sk-...
```

## Running a Real Example

```bash
python src/interfaces/cli/demo.py sample.mp4 \
    --stt \
    --translate \
    --subtitle \
    --speech-provider faster-whisper \
    --translation-provider openai \
    --whisper-model base
```

Flags:
- `--video-provider` — `mock` or `ffmpeg` (default from config: `ffmpeg`)
- `--speech-provider` — `mock` or `faster-whisper` (default: `mock`)
- `--translation-provider` — `mock` or `openai` (default: `mock`)
- `--whisper-model` — `tiny`, `base`, `small`, `medium` (default: `base`)
- `--output-dir` — output directory (default: `output`)
- `--keep-audio` — keep the extracted `audio.wav` file
- `--source-lang` — `zh`, `en`, `auto` (default: `zh`)
- `--target-lang` — target language (default: `vi`)

## Expected Outputs

After successful run, the `--output-dir` (default: `output/`) will contain:

```
output/
├── audio.wav          # Extracted 16kHz mono WAV
├── transcript.json    # Original speech segments with timestamps
├── translated.json    # Segments with translated_text populated
└── subtitle.srt       # SRT subtitle file (translated)
```

### Example `transcript.json`
```json
[
  {
    "id": "speech_0",
    "start": 0.0,
    "end": 3.4,
    "text": "大家好，欢迎来到我的频道。",
    "translated_text": "",
    "source": "speech",
    "confidence": 0.98
  }
]
```

### Example `translated.json`
```json
[
  {
    "id": "speech_0",
    "start": 0.0,
    "end": 3.4,
    "text": "大家好，欢迎来到我的频道。",
    "translated_text": "Chào mọi người, chào mừng đến với kênh của tôi.",
    "source": "speech",
    "confidence": 0.98
  }
]
```

### Example `subtitle.srt`
```srt
1
00:00:00,000 --> 00:00:03,400
Chào mọi người, chào mừng đến với kênh của tôi.
```

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `ProviderError: FFmpeg is not installed` | FFmpeg not in PATH | Install FFmpeg and restart terminal |
| `ProviderError: OPENAI_API_KEY environment variable is not set` | Missing API key | Set `OPENAI_API_KEY` env var |
| `ProviderError: faster-whisper is not installed` | Missing package | `pip install faster-whisper` |
| `FileNotFoundError: [Errno 2] No such file or directory: 'audio.wav'` | Audio extraction failed | Check FFmpeg can read input video format |
| `ModuleNotFoundError: No module named 'openai'` | Missing package | `pip install openai` |
| `ModuleNotFoundError: No module named 'yaml'` | Missing package | `pip install pyyaml` |
| Translation returns empty/incomplete | OpenAI API rate limit or quota | Check API usage; reduce `--whisper-model` size or retry |

## Log Output

Logs are written to stderr (so stdout remains clean JSON):

```
[ExtractAudio] START
[ExtractAudio] DONE (1.2s)

[Speech] START
[Speech] DONE (18.4s)

[Translate] START
[Translate] DONE (2.1s)

[Subtitle] START
[Subtitle] DONE (0.1s)
```

## Configuration File

`config/providers.yaml` controls default providers and model sizes:

```yaml
providers:
  video: ffmpeg
  speech: mock
  ocr: mock
  translation: mock
  subtitle: mock

whisper:
  model_size: base

openai:
  model: gpt-4o-mini
  temperature: 0.3
```

CLI flags override config file values.

## Architecture Note

All new components follow Clean Architecture:
- `FFmpegVideoProvider` in `src/infrastructure/providers/video/`
- `OpenAITranslationProvider` in `src/infrastructure/providers/translation/`
- Logging in `src/shared/logging.py`
- Orchestrator updates in `src/application/orchestrator/job_orchestrator.py`
- CLI composition in `src/interfaces/cli/demo.py`

---

## Stabilized Internal Framework

This milestone introduces four internal framework components that centralize provider wiring, output paths, execution state tracking, and artifact management.

### ProviderRegistry (`src/infrastructure/registry/`)

Maintains a mapping from provider category and name to concrete provider classes.

```python
registry = ProviderRegistry()
provider_cls = registry.get("speech", "faster-whisper")  # Returns FasterWhisperSpeechProvider
```

### ProviderFactory (`src/infrastructure/factory/`)

Creates provider instances from `config/providers.yaml` with support for CLI overrides.

```python
factory = ProviderFactory(config=load_config())
speech_p = factory.create_speech_provider(
    name="faster-whisper",
    overrides={"model_size": "tiny"}
)
```

### ArtifactManager (`src/infrastructure/artifacts/`)

Owns all output paths and file I/O, preventing the orchestrator from hard-coding paths.

```python
am = ArtifactManager(base_dir="output")
am.audio_path("job_123")      # output/job_123/audio/audio.wav
am.transcript_path("job_123") # output/job_123/transcripts/transcript.json
am.translated_path("job_123") # output/job_123/transcripts/translated.json
am.subtitle_path("job_123")   # output/job_123/subtitles/subtitle.srt
am.video_path("job_123")      # output/job_123/videos/output.mp4
```

### JobStateMachine (`src/application/orchestrator/`)

Tracks job execution state with predefined transitions (`created → running → completed/failed`).

### New Output Directory Layout

```text
output/<job_id>/
├── audio/audio.wav
├── transcripts/transcript.json
├── transcripts/translated.json
├── subtitles/subtitle.srt
├── videos/output.mp4
└── logs/job.log
```

### CLI Provider Selection Flow

```text
CLI args
    ↓
Config loader (config/providers.yaml)
    ↓
ProviderFactory → creates concrete provider instances
    ↓
Use cases → receive providers via dependency injection
    ↓
JobOrchestrator → executes pipeline using ArtifactManager and JobStateMachine
```


---

## OCR Framework

OCR is now a first-class optional subsystem parallel to Speech-to-Text.

```text
Video
  ├── Audio Extraction → Speech Segments
  └── Frame Extraction → OCR Segments
           ↓
      MergeSegmentsUseCase
           ↓
        Translation
           ↓
       Subtitle
```

### Frame Extraction

Frame extraction is handled by `FrameExtractionProvider`, not by OCR providers. Frames are stored as file paths, not OpenCV objects.

Supported providers:

- `mock`
- `ffmpeg`

CLI flags:

```bash
--frame-provider mock|ffmpeg
--frame-interval 0.5|1|2
```

### OCR Providers

OCR providers receive extracted `Frame` objects and return `Segment` objects with `source="ocr"`.

Supported providers:

- `mock`
- `paddle`

PaddleOCR stays isolated in Infrastructure; Domain/Application never import `paddleocr`.

### Merge Strategy

`MergeSegmentsUseCase` merges speech and OCR segments by timestamp using stable ordering. No deduplication or text tracking is performed yet.

### Artifact Layout

OCR frames are written under each job output folder:

```text
output/<job_id>/
├── frames/frame_0000.jpg
├── audio/audio.wav
├── transcripts/transcript.json
├── transcripts/translated.json
└── subtitles/subtitle.srt
```

### Mock OCR Example

```bash
python src/interfaces/cli/demo.py sample.mp4   --stt   --ocr   --translate   --subtitle   --video-provider mock   --speech-provider mock   --frame-provider mock   --ocr-provider mock   --translation-provider mock
```

### Paddle OCR Example

```bash
python src/interfaces/cli/demo.py sample.mp4   --ocr   --translate   --subtitle   --video-provider ffmpeg   --frame-provider ffmpeg   --ocr-provider paddle   --translation-provider openai   --frame-interval 1
```


---

## Subtitle Quality Layer

Before subtitle generation, merged and optionally translated segments pass through `NormalizeSegmentsUseCase`.

Responsibilities:

- remove empty segments
- trim whitespace
- sort by timestamp
- deduplicate repeated text
- prefer speech over OCR when overlapping text is similar
- clamp subtitle duration
- merge short neighboring segments when safe
- split long text into subtitle-friendly lines

Default quality settings are configured in `config/providers.yaml` under `subtitle_quality`:

```yaml
subtitle_quality:
  max_chars_per_line: 42
  max_lines_per_subtitle: 2
  min_duration: 0.8
  max_duration: 6.0
  merge_gap_threshold: 0.3
  deduplicate: true
  prefer_speech_over_ocr: true
  remove_empty_segments: true
  trim_whitespace: true
```

CLI quality flags:

```bash
--max-chars-per-line 42
--max-lines-per-subtitle 2
--no-deduplicate
--prefer-ocr-over-speech
```

Normalized artifacts are written to:

```text
output/<job_id>/transcripts/normalized.json
```

Limitations: normalization is rule-based only. No LLM polishing, semantic rewriting, OCR tracking, or video rendering is performed in this layer.


---

## Subtitle Segmentation Layer

This layer splits oversized subtitle segments into multiple subtitle items while preserving all original text content (no content is truncated or discarded).

### Segmentation Rules

- **Split by Length**: If a segment exceeds `max_chars_per_line` or `max_lines_per_subtitle`, it is split.
- **Preserve Text**: All original words remain in identical sequence. No words are dropped.
- **Proportional Timestamps**: Timestamps of the parent segment are divided proportionally among the split child segments according to their character length.
- **Duration Constraints**: Avoids creating subtitle segments shorter than `min_duration` (default 0.8s) or longer than `max_duration` (default 6.0s) when possible.

### Artifacts

The final segmented subtitles are written to:

```text
output/<job_id>/transcripts/segmented.json
```

### CLI Quality Flags

- `--disable-segmentation` — turn off subtitle segmentation
- `--max-duration` — set max duration constraint
- `--min-duration` — set min duration constraint


---

### Video Render Engine

Bổ sung bộ công cụ chèn cứng (burn-in) phụ đề vào video sử dụng FFmpeg thông qua tiến trình `JobOrchestrator` khi bật flag `--render`.

#### Tính năng & Cấu hình:

- **Flag hỗ trợ**:
  - `--render`: Bật tính năng render video.
  - `--render-output`: Ghi đè đường dẫn xuất video đầu ra tùy chọn.
  - `--overwrite`: Cho phép ghi đè file video output đã tồn tại từ trước.
- **Provider cấu hình**: `ffmpeg` (mặc định) và `mock` (dùng cho môi trường thử nghiệm/CI không cài FFmpeg).
- **Yêu cầu hệ thống**: FFmpeg cần phải được cài đặt và cấu hình sẵn trong biến môi trường `PATH`.
- **Đường dẫn artifact**: `output/<job_id>/videos/output.mp4`

#### Hạn chế & Kế hoạch tương lai:
- Phụ đề hiện tại được chèn cứng dưới dạng SRT cơ bản.
- Sẽ hỗ trợ định dạng ASS (Advanced SubStation Alpha) để hỗ trợ căn lề, đổi font chữ và thiết kế style nâng cao trong tương lai.


---

## Subtitle Styling Framework

The project now supports advanced subtitle styling via ASS (Advanced SubStation Alpha) format in addition to SRT.

### Features

- **SubtitleStyle Domain Entity** (`src/domain/entities/subtitle_style.py`):
  Configurable font, size, color, outline, shadow, alignment, margins, bold, italic, spacing, line height, background type, and opacity.

- **Built-in Style Presets** (`src/domain/entities/subtitle_style_preset.py`):
  - `default` — clean standard subtitles
  - `youtube` — bold, outlined, slightly larger
  - `tiktok` — large, bold, heavy outline, higher vertical margin for safe zone
  - `netflix` — yellow text, standard size
  - `minimal` — small, subtle outline, italic

- **ASS Generator Service** (`src/application/services/ass_generator.py`):
  Pure generation service that takes `SubtitleStyle` + segments → ASS content string. No rendering, no filesystem, no FFmpeg.

- **SubtitleProvider Interface** (`src/application/ports/subtitle_provider.py`):
  Supports both `generate_srt` (legacy) and `generate_ass(segments, style)` methods.

- **ASSSubtitleProvider** (`src/infrastructure/providers/subtitle/ass_subtitle_provider.py`):
  Infrastructure implementation using the ASS Generator service.

- **Artifact Paths**:
  Both formats coexist: `output/<job_id>/subtitles/subtitle.srt` and `output/<job_id>/subtitles/subtitle.ass`.

- **FFmpeg Render Auto-Selection**:
  `FFmpegVideoProvider.render_video` detects `subtitle.ass` alongside `subtitle.srt` and prefers the ASS filter (`-vf ass=subtitle.ass`) over the SRT filter (`-vf subtitles=subtitle.srt`). Fallback is automatic.

### CLI Flags

| Flag | Description |
|---|---|
| `--subtitle-format` | `srt` (default) or `ass` |
| `--subtitle-style` | `default`, `youtube`, `tiktok`, `netflix`, `minimal`, or custom from config |
| `--font` | Font family override |
| `--font-size` | Font size override |
| `--outline` | Outline width override |
| `--shadow` | Shadow offset override |
| `--margin-v` | Vertical margin override |

### Example

```bash
python src/interfaces/cli/demo.py video.mp4 \
  --stt --translate --subtitle --render \
  --subtitle-format ass \
  --subtitle-style youtube \
  --video-provider ffmpeg \
  --speech-provider faster-whisper \
  --translation-provider openai
```

### Configuration

Styles are defined in `config/subtitle_styles.yaml`:

```yaml
styles:
  default:
    font_family: Arial
    font_size: 24
    primary_color: "#FFFFFF"
    outline_color: "#000000"
    ...
  youtube:
    font_family: Arial
    font_size: 28
    bold: true
    ...
```

Custom presets can be added to this file and referenced by name via `--subtitle-style`.

### Architecture

```
Segments → SubtitleStyle → ASSGenerator → ASS Content
                              ↓
                     SubtitleProvider (generate_ass)
                              ↓
                     ArtifactManager (writes subtitle.ass)
                              ↓
                     FFmpegVideoProvider (prefers ass filter)
```

All styling logic lives in **Application/Domain**. Infrastructure only handles generation via provider and rendering via FFmpeg. No FFmpeg-specific objects leak into Domain.


---

### Application Composition Root

To simplify dependency resolution and prevent code duplication, we introduced a single **Composition Root** that manages all dependency injection.

#### Components:

- **ApplicationContext** (`src/interfaces/composition/application_context.py`):
  Stores immutable shared instances (config, factory, registries, artifact managers) for the lifespan of the application.
- **ApplicationBuilder** (`src/interfaces/composition/application_builder.py`):
  Initializes config files, merges presets with CLI/API overrides, resolves ports into providers, constructs use cases, and builds the orchestrator.
- **MediaFlowApplication** (`src/interfaces/application/media_flow_application.py`):
  The unified public entry point. Exposes `run(job)`, `run_video(...)`, and `run_cli(args)`.
- **OrchestratorDependencies** (`src/application/orchestrator/orchestrator_dependencies.py`):
  A clean container object wrapping all 16 pipeline dependencies down to a single parameter for `JobOrchestrator`.

#### public entry point usage:

```python
from src.interfaces.composition.application_builder import ApplicationBuilder

# Build the app context and orchestrator graph
builder = ApplicationBuilder(output_dir="my_output")
app = builder.build()

# Run a job
app.run_video(video_path="sample.mp4", speech_to_text=True, render_video=True)
```


---

### Configuration Consolidation & Provider Registration

We unified and consolidated configurations and introduced a decoupled, factory-callback registration mechanism for providers to avoid hard-coded branches.

#### Configuration Ownership:
- `providers.yaml`: Houses provider selection defaults and model specific configurations.
- `subtitle_styles.yaml`: Restricts strictly to visual style presets only.
- `ConfigLoader` (`src/interfaces/config/config_loader.py`): Performs loading, default merging, validation, and CLI override binding to return a typed `AppConfig` model.
- `AppConfig` (`src/interfaces/config/app_config.py`): Dataclass holding defaults, provider configurations, subtitle quality config, and style configurations.

#### Provider Auto Registration:
- Registry mapping category -> name -> class is decoupled into dynamic factory functions registered at module initialization.
- `ProviderRegistry.create(category, name, options)` instantiates providers dynamically.
- `ProviderFactory` delegates entirely to `registry.create(...)`, removing all provider-specific class coupling and conditional creation blocks.

## YOLO burn-in detection

Model **YOLO26n fine-tune** (4 lớp) được dùng để phát hiện burn-in text trên
video. Chi tiết đầy đủ xem [`docs/YOLO_INTEGRATION.md`](docs/YOLO_INTEGRATION.md).

- **4 class**: `subtitle` (0), `watermark` (1), `logo` (2), `other_burn_in_text` (3).
- **Vị trí model**: `models/yolo/best.pt`.
- **Resolve path** (env > config > default):
  - `MEDIA_FLOW_YOLO_WEIGHTS` (biến môi trường),
  - `weights:` trong `config/detection.yaml`,
  - mặc định `models/yolo/best.pt`.
- **Config**: `config/detection.yaml` (device, imgsz, iou, threshold theo class,
  temporal stabilization).
- Model file bị gitignore (`*.pt`), tuyệt đối không commit.

Ví dụ nhanh:

```bash
# Verify model
python scripts/verify_yolo_model.py --device cpu

# Test trên video thật (giữ audio, xuất JSONL)
python scripts/test_video_detection.py \
  --input input/sample5.mp4 \
  --output outputs/detected.mp4 \
  --show-labels --save-json --device cuda
```

Nếu thiếu `best.pt` tại `models/yolo/`, chương trình dừng với hướng dẫn rõ ràng.
Các test YOLO sẽ tự skip khi model không tồn tại.
