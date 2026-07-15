# Chiến lược chuyển đổi OCR và ASR sang Gemini API Free Tier

**Tài liệu kiến trúc — chiến lược triển khai và qualification**

> **Path**: `docs/GEMINI_FREE_TIER_OCR_ASR_STRATEGY.md`
> **Trạng thái**: ACCEPTED STRATEGY — implementation pending
> **Production Ready**: NO

---

## 1. Mục tiêu

Chiến lược này định hướng thay thế hai provider hiện tại bằng các dịch vụ Gemini API trong khi giữ nguyên kiến trúc Clean Architecture và Provider Pattern của dự án.

### Mục tiêu cụ thể

| Thành phần hiện tại | Thay thế bằng | Provider mới |
|---|---|---|
| PaddleOCR | Gemini Vision API | `GeminiVisionOCRProvider` |
| Faster-Whisper | Gemini Audio API | `GeminiAudioSpeechProvider` |
| Gemini TranslationProvider | Giữ nguyên (độc lập) | `GeminiTranslationProvider` |

### Nguyên tắc bắt buộc

- **Ưu tiên Gemini Free Tier** nhằm tối ưu chi phí trong giai đoạn qualification.
- **Giữ nguyên Clean Architecture và Provider Pattern** — domain và application không biết đến Gemini.
- **Không gộp OCR, ASR và translation** thành một provider hoặc một business step.
- **Giữ fail-closed safety** cho dynamic subtitle replacement — các quyết định của downstream pipeline (person overlap, scene risk, watermark) không bị ghi đè bởi Gemini.
- **Đây là chiến lược triển khai và qualification** — chưa phải quyết định Production Ready.

---

## 2. Kiến trúc đích

### Pipeline xử lý video

```text
Video
├── FFmpeg frame extraction
│   └── GeminiVisionOCRProvider (src/infrastructure/providers/ocr/)
│       └── OCR segments + subtitle bbox
│
├── FFmpeg audio extraction
│   └── GeminiAudioSpeechProvider (src/infrastructure/providers/speech/)
│       └── speech segments + timestamps
│
└── OCR / ASR segments (merged bằng MergeSegmentsUseCase)
    └── GeminiTranslationProvider (src/infrastructure/providers/translation/)
        └── Vietnamese target text
            └── NormalizeSegmentsUseCase
                └── SubtitleSegmentationUseCase
                    └── GenerateSubtitleUseCase (ASS/SRT)
                        └── RenderVideoUseCase (FFmpeg)
```

### Nguyên tắc kiến trúc

- **FFmpeg** chịu trách nhiệm extract frame và audio — độc lập với Gemini.
- **GeminiVisionOCRProvider** chỉ nhận `Frame` entities và trả về `Segment` entities với `source="ocr"`.
- **GeminiAudioSpeechProvider** chỉ nhận `audio_path` và trả về `Segment` entities với `source="speech"`.
- **GeminiTranslationProvider** chỉ nhận `Segment` entities và trả về `Segment` entities với `translated_text`.
- **Timeline mapping, safety, layout và ASS renderer** không phụ thuộc Gemini.

### Các port hiện tại (không đổi)

| Port | Method |
|---|---|
| `OCRProvider` | `detect_text(frames: List[Frame]) -> List[Segment]` |
| `SpeechProvider` | `transcribe(audio_path: str, source_language: str) -> List[Segment]` |
| `TranslationProvider` | `translate(segments: List[Segment], target_language: str) -> List[Segment]` |

---

## 3. Quyết định về quota và RPM

### Nguyên tắc

- **Không hardcode quota chính thức của Google** trong mã nguồn.
- RPM, concurrency, retry và backoff là **client-side safety caps**, không đồng nghĩa quota thật.
- Cấu hình phải nằm trong file config (`providers.yaml`) và cho phép **override qua environment variables**.
- OCR, ASR và translation **dùng chung một request scheduler** theo project/model — không tạo limiter riêng lẻ (nguy cơ cộng dồn vượt quota project).

### Cấu hình đề xuất trong cấu hình (conceptual)

```yaml
gemini:
  scheduler:
    rpm_cap: 4
    max_concurrency: 1
    max_retries: 5
    backoff_initial_seconds: 2
    backoff_max_seconds: 60
    respect_retry_after: true
```

### Các chỉ số scheduler phải theo dõi

| Metric | Ghi chú |
|---|---|
| RPM | Requests Per Minute |
| TPM | Tokens Per Minute |
| RPD | Requests Per Day |
| 429 count | Số lần nhận rate limit |
| retry count | Số lần thử lại |
| input tokens | Token đầu vào |
| output tokens | Token đầu ra |

### Lưu ý

- `rpm_cap` không phải là quota thật của Google. Quota thật phải kiểm tra trong **Google AI Studio** khi triển khai.
- Nếu tạo limiter riêng cho từng provider, tổng request từ OCR + ASR + translation có thể vượt quota project dù mỗi limiter riêng vẫn dưới ngưỡng.

---

## 4. Chi phí quota khi tách ba request

Tách OCR, ASR và translation **không đồng nghĩa gửi toàn bộ media ba lần**.

### Trường hợp tối ưu cho video dưới 5 phút

```text
1 ASR request  → toàn bộ audio track (WAV)
1 OCR request  → batch frame đã chọn
1 translation request → toàn bộ text segments
≈ 3 request / video
```

### Khi nào số request tăng

| Nguyên nhân | Ghi chú |
|---|---|
| OCR crop retry | Khi full-frame không rõ ràng |
| Audio request thất bại validation | Transcript không đạt kiểm tra |
| Transcript quá dài | Vượt token limit thực tế |
| API timeout | Cần chunk hoặc retry |
| Quota buộc phải chunk | Audio dài hơn scope hiện tại (>5 phút) |
| Translation batch quá lớn | Vượt context window |

### Công thức tính

```text
total_requests =
    asr_requests
  + ocr_batch_requests
  + translation_batch_requests
  + retry_requests
```

### Phân bổ token

- **Audio** chỉ gửi cho ASR — không gửi cho OCR hay translation.
- **Images** chỉ gửi cho OCR — không gửi cho ASR hay translation.
- **Translation** chỉ nhận text — không gửi media.
- Chi phí tăng thêm chủ yếu từ **prompt, schema và text translation input**.
- Mỗi request phải ghi **usage metadata**.
- Khuyến nghị **token counting** trước khi gửi request lớn.

---

## 5. Chiến lược Gemini ASR

### Mặc định cho video dưới 5 phút

```text
FFmpeg extract mono WAV → one full-audio request to gemini-3.1-flash-lite → structured transcript validation
```

Áp dụng cho audio tối đa 300 giây. Không chunk mặc định: tiết kiệm RPM/RPD, giữ ngữ cảnh, tránh mất chữ tại boundary và không cần deduplicate overlap.

Chỉ chunk khi timeout lặp lại, structured response invalid sau retry, timestamp drift, output bị cắt, vượt token/quota thực tế, hoặc audio dài hơn 300 giây. Khi cần: `60–120 seconds/chunk`, `1–2 seconds overlap`, ưu tiên silence/VAD boundary. Không mặc định chunk 20–40 giây.

### Prompt và validation ASR

```text
Transcribe audible speech verbatim.
Do not translate.
Do not summarize.
Do not correct grammar, names, idioms, or apparent mistakes.
Return timestamps relative to the audio.
Do not include visual subtitles or on-screen text.
Return no segment for music or silence.
Mark unclear speech explicitly.
```

```json
{"source_language":"zh","segments":[{"start":1.25,"end":3.80,"speaker":"SPEAKER_1","text_verbatim":"因果循环","speech_status":"clear"}]}
```

Bắt buộc: `end > start`, timestamp tăng dần, text không rỗng/không duplicate, không summary, không prompt leakage, không tự dịch, language hợp lệ và provenance đầy đủ.

### ASR escalation

```text
primary full-audio invalid or semantically uncertain
→ retry primary once with stricter schema
→ escalation only for failed/uncertain audio region
→ never resend whole audio to escalation unless explicitly configured
```

---

## 6. Chiến lược Gemini OCR không dùng OCR local

```text
sample frames at 1–2 FPS
→ batch full frames
→ primary Gemini OCR
→ temporal consensus
→ accepted?
   ├── yes → produce OCR interval
   └── no → crop locally from Gemini bbox → upscale → retry primary once
             → still ambiguous? escalation enabled: Gemini 3.5 Flash; otherwise: REVIEW_REQUIRED
```

Gemini full-frame pass trả source text, subtitle classification và bbox:

```json
{"frame_id":"frame_0012","timestamp":4.0,"detections":[{"text_verbatim":"原文字幕","language":"zh","text_type":"subtitle","box_2d":[760,180,840,820],"transcription_status":"readable"}]}
```

`box_2d` dùng `[ymin, xmin, ymax, xmax]` trên thang 1000 và provider phải convert sang pixel trước Application layer. Prompt bắt buộc transcribe đúng ký tự hiển thị, không translate/correct/infer, không trộn subtitle với watermark/UI/logo/scene text và trả `detections: []` khi không có subtitle.

Crop local chỉ làm pixel crop, resize và optional contrast normalization. Không dùng PaddleOCR/OCR local để xác định crop. PaddleOCR chỉ giữ cho benchmark, shadow comparison hoặc diagnostic fallback được yêu cầu rõ, không tham gia auto-replacement sau khi Gemini provider được chọn.

Crop retry chỉ cho text nhỏ, bbox jitter, disagreement, nền phức tạp, bubble caption, partial occlusion hoặc `transcription_status` không rõ. Không crop mọi frame.

---

## 7. Temporal consensus và safety cho OCR

Không tin LLM-generated confidence. Suy ra confidence từ:

```text
text_consensus_score
bbox_iou_consensus
bbox_center_jitter
temporal_presence_ratio
subtitle_classification_agreement
```

Auto-accept chỉ khi có detection nhiều frame, text/bbox/classification đồng thuận, geometry hợp lệ, và không có watermark/scene/person hard risk. Bbox cuối dùng median hoặc controlled union, không dùng mù bbox một frame.

Gemini OCR không ghi đè downstream safety. `sample5 interval_6` phải giữ `SKIPPED` và `true_person_overlap_risk`.

---

## 8. Tách OCR, ASR và translation

```text
GeminiVisionOCRProvider
GeminiAudioSpeechProvider
GeminiTranslationProvider
```

Không xây `GeminiVideoUnderstandingProvider` gộp OCR + ASR + translate + replacement. Tách riêng bảo đảm provenance, retry/cache/benchmark/model routing riêng, phân biệt lỗi và ngăn ASR text bị map vào OCR bbox.

Mỗi output lưu provenance tối thiểu:

```json
{"provider":"gemini","model":"gemini-3.1-flash-lite","model_role":"primary","model_stage":"STABLE","retirement_time":"...","model_validation_status":"available","api_surface":"configured_api_surface","request_id":"...","cache_hit":false,"prompt_version":"gemini_ocr_v1","schema_version":"ocr_detection_v1","source_kind":"image","media_sha256":"abc123...","retry_count":0,"input_tokens":1423,"output_tokens":89}
```

`api_surface` là field đề xuất, vì kiến trúc chưa chốt GenerateContent hay Interactions API.

---

## 9. Model routing và Free Tier chính xác

### Routing cuối cùng

```text
Primary:      gemini-3.1-flash-lite
Escalation:   gemini-3.5-flash
Legacy fallback: none
Local diagnostic providers: PaddleOCR, Faster-Whisper
```

Không dùng `gemini-2.5-flash`, `gemini-2.5-flash-lite`, model preview hoặc alias động (`gemini-flash-latest`) làm default/fallback runtime. Luôn dùng exact stable model ID vì alias có thể đổi model mà không đổi config/cache key.

### Free Tier matrix (verified 2026-07-15)

| API surface | `gemini-3.1-flash-lite` | `gemini-3.5-flash` |
|---|---|---|
| Standard API | Free Tier input/output available | Free Tier input/output available |
| Batch API | Free Tier available | Free Tier not available |

media-flow runtime chỉ dùng **Standard API**. Batch API chỉ dùng cho benchmark hoặc offline khi: model có Free Tier Batch (chỉ `gemini-3.1-flash-lite`), turnaround chậm được chấp nhận, job không cần kết quả ngay và quota Batch đã kiểm tra. Với `gemini-3.5-flash`, không dùng Batch trong chiến lược Free Tier.

Free Tier pricing eligibility không bảo đảm request chạy. Phải phân biệt độc lập:

```text
pricing eligibility | project model access | active quota | actual capacity
```

### Cấu hình conceptual cuối cùng

> `ConfigLoader` hiện chưa hỗ trợ environment expansion — đây là conceptual example, không dùng trực tiếp làm providers.yaml.

```yaml
gemini:
  free_tier_only: true
  scheduler:
    rpm_cap: 4
    max_concurrency: 1
    max_retries: 5
    backoff_initial_seconds: 2
    backoff_max_seconds: 60
    respect_retry_after: true
  routing:
    primary_model: gemini-3.1-flash-lite
    escalation:
      enabled: true
      model: gemini-3.5-flash
      max_requests_per_job: 2
      require_runtime_availability: true
      require_active_project_quota: true
providers:
  ocr:
    gemini_vision:
      model_role: primary
      crop_retry_enabled: true
      escalation_enabled: true
  speech:
    gemini_audio:
      model_role: primary
      full_audio_max_seconds: 300
      chunk_on_validation_failure: true
      escalation_enabled: true
  translation:
    gemini:
      model_role: primary
      batch_segments: true
      escalation_enabled: true
```

`rpm_cap: 4` chỉ là client-side safety cap ví dụ, không phải quota chính thức. OCR/ASR/translation dùng cùng scheduler. `max_requests_per_job` giới hạn tiêu thụ escalation quota.

---

## 10. Chiến lược escalation tối ưu

```text
request primary model
→ validate structured output
→ output valid?
   ├── yes → accept
   └── no
       → error can improve with same model?
           ├── yes → one primary retry with stricter prompt/schema
           │        → valid? yes → accept | no → consider escalation
           └── no  → consider escalation
```

Chỉ gọi `gemini-3.5-flash` khi: escalation bật; model tồn tại/khả dụng; active quota đã check; không phải hard safety reject; lỗi thuộc semantic ambiguity hoặc transcription uncertainty; chưa vượt request budget của job.

```text
escalation valid → accept với provenance model_role=escalation
escalation invalid hoặc mâu thuẫn với primary → REVIEW_REQUIRED
```

Không gọi escalation khi: API key sai; model không tồn tại; bbox ngoài frame; audio file hỏng; input thiếu; schema nội bộ sai; person/watermark/scene safety reject; quota budget đã hết. Không dùng model mạnh để ghi đè hard safety.

---

## 11. Cache strategy

Cache key bắt buộc bao gồm:

```text
input bytes or normalized input text
model exact ID
model role
prompt version
schema version
generation config
media resolution
API surface
```

Ví dụ:

```text
SHA256(
  normalized_input
  + exact_model_id
  + model_role
  + prompt_version
  + schema_version
  + generation_config
  + media_resolution
  + api_surface
)
```

Primary và escalation phải có cache namespace khác nhau. Không dùng cache của `gemini-3.1-flash-lite` cho `gemini-3.5-flash`.

---

## 12. Proposed provider error taxonomy

Chỉ `ProviderError` tồn tại trong `src/shared/exceptions.py`. Các subclass dưới đây là proposed contracts cho Phase B. Until implemented, Gemini infrastructure map mọi lỗi vào `ProviderError`.

| Lỗi Gemini | Proposed subclass | Xử lý |
|---|---|---|
| 429 | `ProviderRateLimitError` | retry có backoff |
| Timeout | `ProviderTimeoutError` | retry |
| Invalid JSON | `ProviderResponseError` | retry 1 lần prompt chặt hơn |
| Schema mismatch | `ProviderResponseError` | retry 1 lần |
| Empty response | `ProviderResponseError` | retry 1 lần |
| Blocked response | `ProviderBlockedError` | fail closed |
| Upload failure | `ProviderIOError` | retry |
| Invalid timestamps | `ProviderValidationError` | fail closed |
| Invalid bbox | `ProviderValidationError` | fail closed |
| Model unavailable | `ProviderUnavailableError` | retry escalation |
| Model lifecycle/shutdown/not found | `ProviderConfigurationError` | fail fast |

Không tự chế transcript/OCR text, không tạo placeholder.

---

## 13. Bảo mật và Free Tier safety

- API key chỉ lấy từ `$env:GEMINI_API_KEY` hoặc secret store. Không lưu trong config repository.
- Không log raw authorization headers hoặc API keys.
- Free Tier data policy: Nội dung Free Tier có thể được dùng để cải thiện sản phẩm của Google. Không dùng media nhạy cảm trong qualification nếu chưa được phép.
- Cleanup: Xóa uploaded files trên Google servers sau khi job hoàn thành.
- Artifact storage: Không lưu raw media response ngoài artifact policy.
- Free Tier pricing hiện hành không bảo đảm model có capacity. OCR, ASR và translation dùng chung project quota. Không dựa vào Free Tier cho production SLA.

---

## 14. Model lifecycle and availability validation

### Quy tắc

- Không hardcode model ID trong provider. Model lấy từ typed config.
- Cần validation lúc application startup/qualification:
  - Gọi [Models API](https://ai.google.dev/api/models) `models.get` hoặc `models.list` (filter `supportedGenerationMethods` / `supported_actions` chứa `generateContent`).
  - Xác minh primary model tồn tại và đang `STABLE` hoặc `PREVIEW` (không phải `LEGACY`, `DEPRECATED` hoặc `RETIRED`).
  - Xác minh escalation model tồn tại nếu được cấu hình.
  - Xác minh model hỗ trợ `generateContent`. Fail fast nếu model đã shutdown hoặc không khả dụng.
- Không tự động đổi model mà không ghi provenance.
- Mọi thay đổi model phải làm invalid cache key.
- Benchmark report phải ghi model ID chính xác, không chỉ ghi "Flash-Lite".
- Google states that a deprecation-table shutdown date for an active model is the earliest date on which retirement may occur. The exact shutdown date may be communicated later.

### Pseudo-flow validation

```text
load configured model (from typed config)
→ query Models API (models.get or models.list)
→ available and supports generateContent?
   ├── yes → check modelStatus.modelStage at generation time (GenerateContentResponse)
   │         ├── STABLE → continue
   │         ├── PREVIEW → warn, continue
   │         ├── LEGACY → fail fast with ProviderConfigurationError
   │         ├── DEPRECATED → fail fast with ProviderConfigurationError
   │         ├── RETIRED → fail fast with ProviderConfigurationError
   │         ├── MODEL_STAGE_UNSPECIFIED → fail fast with ProviderConfigurationError
   │         └── UNSTABLE_EXPERIMENTAL → fail fast with ProviderConfigurationError
   └── no  → fail fast with ProviderConfigurationError
```

### Model lifecycle table (verified 2026-07-15)

| Model | Vai trò | Trạng thái | Lifecycle date | Ý nghĩa | Hành động |
|---|---|---|---|---|---|
| `gemini-2.0-flash-lite` | Historical | Shut down | 2026-06-01 | Actual shutdown date | Không dùng |
| `gemini-2.5-flash` | Legacy candidate | Stable nhưng sắp EOL | 2026-10-16 | Earliest possible shutdown date | Không chọn cho code mới, khuyên dùng `gemini-3.5-flash` |
| `gemini-3.1-flash-lite` | Primary | Stable | 2027-05-07 | Earliest possible shutdown date | Qualification default |
| `gemini-3.5-flash` | Escalation | Stable | Chưa công bố | No shutdown date announced | Chỉ retry trường hợp khó |
| `gemini-3.1-flash-lite-preview` | Historical preview | Shut down | 2026-05-25 | Actual shutdown date | Không dùng |

**Checklist trước release:**

```
[ ] Primary model exists (models.list)
[ ] Escalation model exists (models.list)
[ ] Required input modality supported (models.get)
[ ] Structured output supported (models.get)
[ ] Free Tier/pricing checked
[ ] Active project quota checked
[ ] Deprecation page checked
[ ] Cache namespace includes model ID
[ ] Benchmark records exact model ID
```

---

## 15. Shared request scheduler strategy

- Scheduler được chia sẻ theo API project. Không tạo limiter riêng cho từng provider.
- Rate limits áp dụng theo project, không theo API key. Hạn mức RPD reset nửa đêm giờ Thái Bình Dương (Pacific time).
- Scheduler theo dõi tối thiểu: RPM, TPM, RPD, request count theo model, retry count, escalation count, 429 count, input/output tokens, cache hit/miss.
- Active limits phải check trong Google AI Studio. Scheduler phải hỗ trợ per-model budget và global project budget.

### Pseudo-design:

```text
GeminiRequestScheduler
├── global project request window
├── per-model counters
├── daily request budget (RPD)
├── token budget (TPM)
└── shared by:
    ├── GeminiVisionOCRProvider
    ├── GeminiAudioSpeechProvider
    └── GeminiTranslationProvider
```

---

## 16. Translation strategy cuối cùng

Gemini TranslationProvider tiếp tục độc lập.

### Luồng xử lý (Translation Flow)

```text
validated OCR/ASR source segments
→ batch source text
→ gemini-3.1-flash-lite translation
→ TranslationQualityGate (chống sót CJK text trong bản dịch Việt)
→ accepted?
   ├── yes → continue
   └── no
       → retry primary một lần
       → optional gemini-3.5-flash escalation
       → vẫn invalid → REVIEW_REQUIRED
```

Nghiêm cấm translation provider đọc ảnh, đọc audio, tạo bbox, quyết định thay thế (replacement decision), sửa safety, hoặc tự tạo placeholder.

---

## 17. Quota budget đề xuất cho mỗi video

Với video dưới 5 phút, happy path:
- ASR: 1 request.
- OCR: 1-N batch requests tùy thuộc vào số lượng frame và giới hạn API.
- Translation: 1 batch request.
- Expected minimum: Khoảng 3 request groups/video.

### Retry budget đề xuất:
- Primary schema retry: Tối đa 1 lần cho mỗi operation.
- OCR crop retry: Chỉ chạy trên frame/interval mơ hồ.
- Escalation: Tối đa 2 requests/job mặc định (giới hạn tiêu thụ escalation quota).
- Hard maximum: Có thể cấu hình theo từng job.

---

## 18. Lộ trình triển khai

### Phase A — Documentation và contracts (hiện tại)

- [x] Hoàn thành tài liệu này.
- [ ] Khóa config và provenance contract.
- Chưa đổi code runtime.

### Phase B — Shared Gemini infrastructure

**Status: IMPLEMENTED, qualification infrastructure only.** OCR/ASR providers,
benchmarks, default-provider changes, and production readiness remain out of
scope. See [GEMINI_SHARED_INFRASTRUCTURE_PHASE_B_REPORT.md](GEMINI_SHARED_INFRASTRUCTURE_PHASE_B_REPORT.md).

Implemented at `src/infrastructure/gemini/`:

```text
src/infrastructure/gemini/
├── __init__.py
├── client.py                 # Lazy Standard generateContent REST client
├── contracts.py              # Internal request/response/provenance contracts
├── runtime.py                # Application-scoped shared runtime
├── request_scheduler.py      # Shared RPM/TPM scheduler với token tracking
├── response_cache.py         # Disk-based response cache
├── structured_output.py      # Structured JSON parsing
├── model_validator.py        # Models API validation policy
├── file_manager.py           # Upload/cleanup temporary files
├── usage_recorder.py         # Redaction-safe usage/provenance recording
└── exceptions.py             # ProviderError-compatible Gemini error taxonomy
```

### Phase C — Gemini OCR provider

**Status: IMPLEMENTED, qualification/shadow mode only.** See [GEMINI_VISION_OCR_PHASE_C_REPORT.md](GEMINI_VISION_OCR_PHASE_C_REPORT.md).

- [x] Implement OCRProvider port (GeminiVisionOCRProvider).
- [x] Chạy được shadow/qualification mode và ghi artifact riêng, không ảnh hưởng output.
- [x] Giữ nguyên default provider.

### Phase D — Gemini ASR provider

**Status: COMPLETED, qualification/shadow-only mode.** See
[GEMINI_AUDIO_ASR_PHASE_D_REPORT.md](GEMINI_AUDIO_ASR_PHASE_D_REPORT.md).

- [x] Implement `SpeechProvider` port.
- [x] Mặc định full-audio request cho video dưới 5 phút.
- [x] Chạy shadow mode.

### Phase E — Benchmark
**Status: IMPLEMENTED, offline comparative framework only.** Live qualification
remains blocked/incomplete and production readiness remains NO. See
[GEMINI_PHASE_E_COMPARATIVE_BENCHMARK.md](GEMINI_PHASE_E_COMPARATIVE_BENCHMARK.md).

- [x] Implement offline PaddleOCR versus Gemini full-frame and crop-retry comparison framework.
- [x] Implement offline Faster-Whisper small versus Gemini full-audio comparison framework.
- [x] Preserve `sample5 interval_6` hard safety regression check.
- [x] Write offline benchmark artifacts without provider calls.

### Phase F — Hybrid escalation
- Chỉ retry các frame/audio không đạt validation.
- Dùng escalation model khi pass đầu không đủ confidence.

### Phase G — Chuyển default
- Đổi default provider sau khi benchmark và visual QA đạt acceptance.
- Không xóa PaddleOCR/Faster-Whisper. Giữ lại trong codebase làm fallback.

---

## 19. Acceptance criteria

### OCR
- Subtitle detection recall không giảm so với PaddleOCR baseline.
- OCR text quality được human review.
- Source text hallucination: Zero tolerance.
- Bbox jitter trong ngưỡng cho phép.
- Watermark/scene text false cover: Zero.
- Dynamic replacement safety không giảm — `sample5` hard reject giữ nguyên.

### ASR
- Có reference transcript để đo.
- Đo lường và báo cáo CER/WER và độ lệch timestamp.
- Không bị mất hoặc cắt cụt văn bản ở đầu và cuối tệp âm thanh.
- Không chứa các phân đoạn trùng lặp, tóm tắt hoặc tự suy diễn từ ngữ.
- Kiểm duyệt và ghi nhận các lỗi liên quan đến tên riêng và từ đồng âm.

### End-to-end
- `placeholder_count = 0` (không phụ đề giả lập).
- `source_copy_without_reason = 0` (không sao chép nguyên văn bản gốc mà không có lý do).
- `untranslated_cjk_in_vietnamese = 0` (không còn chữ Trung/Nhật/Hàn sót lại trong bản dịch Việt).
- Provenance: OCR, ASR, translation đầy đủ.
- Hard safety reject: Giữ nguyên.
- Source leakage review được thực hiện.
- `production_ready` bắt buộc phải là `NO` cho đến khi qualification hoàn tất.

---

## 20. Decision record

- **Strategy accepted**: 2026-07-15.
- **Model correction**:
  - Removed `gemini-2.0-flash-lite` as active default (shut down 2026-06-01).
  - Removed `gemini-2.5-flash` as the escalation candidate for new implementation because it is a valid but aging stable model with an announced earliest possible shutdown date of 2026-10-16. Its recommended replacement is `gemini-3.5-flash`.
  - Selected `gemini-3.1-flash-lite` (stable, GA) as primary qualification model.
  - Selected `gemini-3.5-flash` (stable, GA) as configurable escalation candidate (Standard API only for Free Tier strategy).
  - Added runtime model availability validation requirement.
  - Added cache invalidation policy on model change.
- **Implementation status**:
  - Provider code not yet implemented.
  - Scheduler not yet implemented.
  - Exception subclasses not yet implemented (only `ProviderError` exists).
  - Benchmark not yet executed.
  - Production readiness remains **NO**.

---

## 21. Official Gemini sources

Tài liệu được xác minh từ các nguồn chính thức của Google AI for Developers (truy cập 2026-07-15):

| Source | URL | Nội dung xác minh |
|---|---|---|
| Gemini API model deprecations | https://ai.google.dev/gemini-api/docs/deprecations | Danh sách model đã/gần shutdown |
| Gemini 3.1 Flash-Lite model page | https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite | Modality, limits, capabilities |
| Gemini 3.5 Flash model page | https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash | Modality, limits, capabilities |
| Gemini API pricing | https://ai.google.dev/gemini-api/docs/pricing | Free Tier, rate limits, pay-as-you-go |
| Models API reference | https://ai.google.dev/api/models | `models.get`, `models.list`, Model resource |
| Generate Content API (ModelStatus) | https://ai.google.dev/api/generate-content | `modelStage`, `retirementTime` |
| Rate limits | https://ai.google.dev/gemini-api/docs/rate-limits | Cấu trúc rate limit theo project/IP |

---

*Tài liệu tham khảo nội bộ: `PROJECT_OPERATING_SPECIFICATION.md`, `FRAMEWORK_V1_SIGNOFF.md`, `ROADMAP_AFTER_V1.md`, `ASR_BENCHMARK_REPORT.md`, `REAL_OUTPUT_QUALITY_REPORT.md`, `static_source_cover_mvp_baseline.md`, `static_source_cover_mvp_v2_visual_qa.md`, `phase9_v5_replay_audit.md`, `phase9b_same_source_baseline.md`, `phase9g_local_safety_audit.md`, `project_reassessment_phase8.md`, `resource_policy.md`*
