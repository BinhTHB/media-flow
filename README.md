# MediaFlow

Hệ thống dịch/dub video short (Trung/Anh → Việt) theo kiến trúc **Feature-first Modular Clean Architecture**.

Tài liệu kiến trúc đầy đủ nằm trong [`docs/`](docs/README.md). File này chỉ hướng dẫn vận hành.

## Kiến trúc thực tế

```text
CLI / Web (localhost:8765)
        │
        ▼
JobRunner            src/interfaces/jobs/runner.py     validate + preflight + compose
        │
        ▼
ProcessVideoUseCase  src/application/process_video.py  orchestration mỏng
        │
        ├── VisualPipeline  src/features/visual/       YOLO → tracker → 4 class → render video-only
        ├── AudioPipeline   src/features/audio/        ASR → translate → TTS → AudioComposer
        └── mux 2 stream → final MP4
```

```text
src/
├── core/                 # BBox, Detection, Track, SubtitleEvent, SpeechSegment, ports (ABC)
├── application/          # ProcessVideoUseCase, ProcessingRequest/Result
├── features/
│   ├── audio/            # AudioPipeline, AudioComposer, ports
│   ├── visual/           # VisualPipeline, tracker, processors/, actions/, subtitle/
│   ├── translation/      # TranslationService dùng chung cho Audio và Subtitle
│   └── publishing/       # PublishService (retry/backoff, idempotency)
├── infrastructure/       # YOLO, Gemini, Groq, OpenRouter, TTS, FFmpeg, stage cache
├── interfaces/           # cli/, web/, jobs/, composition/, publish_worker/
└── shared/               # logging, exceptions
```

Quy tắc phụ thuộc: `core ← features ← application ← interfaces`. Infrastructure implement port của features/application. Chi tiết xem [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Cài đặt

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Yêu cầu hệ thống: **FFmpeg + FFprobe trong PATH**, Python 3.10+.

Model YOLO 4 lớp đặt tại `models/yolo/best.pt` (đã track trong git, ~5 MB). Flow `yolo-text` cần thêm `models/yolo/text/best.pt` tải từ [RoyRud1902/yolo11n-text](https://huggingface.co/RoyRud1902/yolo11n-text).

## Cấu hình API key

Copy `.env.example` → `.env` rồi điền. Key pool nhiều tài khoản đặt trong **một** biến, phân cách bằng dấu phẩy.

| Biến | Dùng cho |
|---|---|
| `GEMINI_API_KEY` | OCR phụ đề, translation fallback, ASR fallback, TTS gemini-live |
| `GROQ_API_KEY` | ASR primary (`whisper-large-v3`) |
| `OPENROUTER_API_KEYS` | Translation pool, ASR correction, duration rewrite, Fish TTS |
| `ELEVENLABS_API_KEYS` | Eleven v3 TTS + Dubbing v2 |

`.env` được load tự động bởi web server, không ghi đè biến môi trường đã có sẵn.

Đổi model Gemini: `MEDIAFLOW_GEMINI_MODEL=<model>` trong `.env`.

## Giao diện web

```powershell
.venv\Scripts\python.exe -m src.interfaces.web.server
```

Mở `http://127.0.0.1:8765` — mặc định vào AI Studio (`studio.html`). Dubbing Studio ở `/dubbing.html`.

- Upload lưu tại `input/uploads/`, output tại `outputs/`.
- Server chỉ bind localhost, không gọi shell.
- Tiến trình stream log realtime vào terminal của job.

## CLI

```powershell
# Dịch + thay phụ đề, giữ logo/watermark
.venv\Scripts\python.exe -m src.interfaces.cli --input video.mp4 `
    --preset yolo-4class --subtitle-action translate `
    --extra-text-action preserve --overwrite

# Lồng tiếng AI + blur phụ đề gốc
.venv\Scripts\python.exe -m src.interfaces.cli --input video.mp4 `
    --subtitle-action blur --dub-audio --voice-id elevenlabs:env
```

### Flags chính

| Flag | Chức năng |
|---|---|
| `--input` | Video đầu vào (bắt buộc) |
| `--output` | Video đầu ra (mặc định `outputs/cli/<tên>-<preset>.mp4`) |
| `--preset` | Flow nhận diện: `yolo-4class` (mặc định) \| `yolo-text` |
| `--subtitle-action` | `translate` \| `blur` (mặc định) \| `preserve` |
| `--dub-audio` | Bật lồng tiếng AI |
| `--extra-text-action` | Logo/watermark/text khác: `blur` (mặc định) \| `preserve` |
| `--source-language` / `--target-language` | Mặc định `auto` / `vi` |
| `--subtitle-layout` | `in-place` \| `temporal` (mặc định) \| `bottom` |
| `--voice-id` | `elevenlabs:env`, `fish:<id>`, `gemini-live:<name>`, `vi-VN-HoaiMyNeural` |
| `--hide-source-subtitle` / `--show-source-subtitle` | Ẩn (mặc định) / giữ sub gốc |
| `--device`, `--profile` | `auto`/`cuda`/`cpu`, `fast`/`balanced`/`quality` |
| `--no-visual-cache`, `--overwrite` | Tắt cache stage / ghi đè output |

### Preset

Preset chỉ chọn **flow nhận diện**. Việc pipeline làm gì với phụ đề/audio/text khác là tuỳ chọn riêng, áp dụng cho cả 2 flow.

| Preset | Model | Thông số |
|---|---|---|
| `yolo-4class` (mặc định) | YOLO tự train `models/yolo/best.pt` (4 lớp) | conf 0.05, imgsz 1024, rescue ON |
| `yolo-text` | `yolo11n-text` 1 lớp `text` + phân loại theo hình học | conf 0.20, imgsz 640, rescue OFF |

Preset cũ (`translate-video`, `translate-subtitle`, `dub-audio`, `blur-all`, `blur-and-dub`, `preserve-all`, suffix `-generic`) vẫn chạy nhưng deprecated — tự map sang flow + tuỳ chọn tương ứng.

Job chỉ giữ nguyên hình khi `--subtitle-action preserve --extra-text-action preserve` (visual pipeline tự tắt, không load YOLO).

`yolo-text` phát hiện **mọi** text trên khung nên tốn API hơn đáng kể. Sửa `text_classifier.py` sẽ tự vô hiệu cache detection.

### Provider routing

- `--asr-provider auto|groq|gemini`: `auto` ưu tiên Groq Whisper Large V3, fallback Gemini Audio.
- `--asr-correction auto|off|openrouter`: chỉ kích hoạt khi transcript có dấu hiệu nghi ngờ; không đổi timestamp.
- `--translation-provider auto|openrouter|gemini`: `auto` ưu tiên OpenRouter pool, escalation Gemini nếu cần.
- `--tts-provider auto|elevenlabs|fish|gemini-live|edge`: `auto` đi ElevenLabs v3 → Fish → Edge. `gemini-live` chỉ dùng explicit vì có thể paraphrase thay vì đọc verbatim.

## Publishing (reup tự động)

Video trong queue → sinh metadata → upload qua API chính thức → ghi state, không bao giờ upload trùng.

```powershell
python -m src.interfaces.publish_worker --queue outputs/publish_queue --target youtube
```

Chi tiết setup OAuth YouTube và sidecar metadata: [`docs/PUBLISHING.md`](docs/PUBLISHING.md).

## Test

```powershell
# Quality gate đầy đủ (compile, secret scan, architecture, pytest)
.venv\Scripts\python.exe scripts\quality.py

# Chỉ pytest
.venv\Scripts\python.exe -m pytest -q
```

Test dùng model YOLO thật là integration test, mặc định skip. Bật rõ ràng:

```powershell
$env:MEDIA_FLOW_RUN_YOLO_INTEGRATION="1"
.venv\Scripts\python.exe -m pytest tests/test_yolo_smoke.py
```

Chi tiết CI, job lifecycle, benchmark policy: [`docs/QUALITY_AND_OPERATIONS.md`](docs/QUALITY_AND_OPERATIONS.md).

## Artifacts

Mỗi job có workspace riêng tại `outputs/jobs/<job_id>/`:

```text
job.json               # request + metadata
result.json            # ProcessingResult
visual/subtitle_events.json   # chỉ khi chạy subtitle processing
audio/segments.json    # chỉ khi chạy audio ASR
logs/
```

Chỉ tạo artifact cho feature được bật. Toàn bộ `outputs/` đã gitignore.

## Troubleshooting

| Lỗi | Nguyên nhân | Cách sửa |
|---|---|---|
| `FFmpeg and FFprobe must be available in PATH` | Chưa cài FFmpeg | `winget install Gyan.FFmpeg`, mở terminal mới |
| `YOLO weights not found` | Thiếu `models/yolo/best.pt` | Kiểm tra `git status models/yolo/` |
| `Translation requires OPENROUTER_API_KEY(S) or GEMINI_API_KEY` | Chưa set key | Điền `.env` |
| `Missing dependency 'ultralytics'` | Sai interpreter | Dùng đúng `.venv\Scripts\python.exe` |
| Subtitle không thay được | OCR/dịch fail | Xem `visual/subtitle_events.json`, status `PARTIAL`/`FAILED` |

## Governance

- [`AGENTS.md`](AGENTS.md) — handbook cho AI coding agent và developer.
- [`docs/PROJECT_OPERATING_SPECIFICATION.md`](docs/PROJECT_OPERATING_SPECIFICATION.md) — technical constitution.
- [`docs/`](docs/README.md) — bộ tài liệu kiến trúc đầy đủ.
