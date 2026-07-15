# Chiến lược chuyển đổi OCR và ASR sang Gemini API Free Tier

**Tài liệu kiến trúc — chiến lược triển khai và qualification**

> **Path**: `docs/GEMINI_FREE_TIER_OCR_ASR_STRATEGY.md`
> **Trạng thái**: DRAFT — chưa triển khai provider
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

### Cấu hình đề xuất trong `providers.yaml`

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
FFmpeg extract audio → mono WAV → gửi toàn bộ audio trong một Gemini ASR request
```

**Không chunk mặc định.**
Lý do:
- Tiết kiệm RPM/RPD.
- Giữ toàn bộ ngữ cảnh hội thoại.
- Không mất chữ ở boundary giữa các chunk.
- Không cần deduplicate overlap region.
- Timestamp dễ quản lý, không bị trôi lệch giữa các chunk.
- Implementation đơn giản hơn.

### Chỉ chunk khi cần

Kích hoạt chunk khi có bằng chứng:
- Request timeout liên tục.
- Transcript JSON không ổn định.
- Timestamp drift nghiêm trọng.
- Vượt quota/token limit thực tế.
- Cần retry một đoạn nhỏ.
- Audio dài hơn scope thiết kế (>5 phút).

Khi phải chunk:

```text
chunk duration: 60–120 giây
overlap: 1–2 giây
ưu tiên cắt tại silence/VAD boundary
```

Không mặc định chunk 20–40 giây.

### Prompt ASR

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

### Structured schema (JSON)

```json
{
  "source_language": "zh",
  "segments": [
    {
      "start": 1.25,
      "end": 3.80,
      "speaker": "SPEAKER_1",
      "text_verbatim": "因果循环",
      "speech_status": "clear"
    }
  ]
}
```

### Validation bắt buộc

| Rule | Mô tả |
|---|---|
| `end > start` | Timestamp hợp lệ |
| Timestamp tăng dần | Không đảo ngược |
| Không text rỗng | `text_verbatim` không rỗng |
| Không duplicate | Không segment trùng lặp |
| Không hallucinated summary | Không tóm tắt |
| Không lẫn prompt instructions | Output chỉ chứa transcript |
| Không tự dịch | Không translate trong ASR |
| Language phù hợp | Ngôn ngữ khớp source_language |
| Provenance đầy đủ | Provider, model, prompt_version, schema_version |

---

## 6. Chiến lược Gemini OCR không dùng OCR local

> **Không cần PaddleOCR hoặc OCR local để xác định crop.**

### Chiến lược chính

```text
Full-frame Gemini pass
→ Gemini trả subtitle text + bbox
→ temporal consensus
→ accept hoặc crop retry
```

### Pass đầu tiên — Full-frame

Gửi full frame theo batch (không crop, không tiền xử lý).
Gemini phải trả:

```json
{
  "frame_id": "frame_0012",
  "timestamp": 4.0,
  "detections": [
    {
      "text_verbatim": "原文字幕",
      "language": "zh",
      "text_type": "subtitle",
      "box_2d": [760, 180, 840, 820],
      "transcription_status": "readable"
    }
  ]
}
```

> **Lưu ý**: Tọa độ `box_2d` tuân theo định dạng Gemini [ymin, xmin, ymax, xmax] trên thang đo 1000. Provider phải convert về tọa độ pixel trước khi trả về Application layer.

### Prompt yêu cầu cho OCR

```text
Transcribe characters exactly as visible.
Do not translate.
Do not correct grammar.
Do not infer missing characters.
Do not merge subtitles with watermarks, UI, logos or scene text.
Return an empty detections list when no subtitle is visible.
```

### Crop retry

Chỉ dùng khi pass đầu không đủ chắc chắn.

**Quy trình:**

```text
Gemini full-frame bbox
→ local pixel crop bằng FFmpeg/OpenCV/Pillow
→ upscale crop
→ gửi crop lại Gemini để đọc text
```

**Local processing chỉ crop và resize pixel — không OCR local.**
Crop retry chỉ kích hoạt khi:
- Text quá nhỏ.
- Bbox jitter giữa các frame.
- Các frame không đồng thuận text.
- Subtitle có nền phức tạp.
- Bubble caption.
- Text bị che một phần.
- `transcription_status` không rõ ràng.

**Không crop mọi frame mặc định** để tránh tăng quota.

### Sampling

```text
Initial sampling: 1–2 FPS
Adaptive sampling: tăng frame quanh thời điểm text thay đổi
```

Không gửi mọi frame video.

---

## 7. Temporal consensus cho OCR

Không tin confidence tự khai của LLM.

### Các chỉ số đồng thuận

| Metric | Mô tả |
|---|---|
| `text_consensus_score` | Đồng thuận nội dung text giữa các frame |
| `bbox_iou_consensus` | IoU trung bình của bbox qua thời gian |
| `bbox_center_jitter` | Độ dịch chuyển tâm bbox |
| `temporal_presence_ratio` | Tỷ lệ frame có detection / tổng frame |
| `subtitle_classification_agreement` | Nhất quán trong phân loại subtitle |

### Auto-accept condition

Một interval chỉ được auto-accept khi đạt:
- Detection xuất hiện ở nhiều frame.
- Text đồng thuận cao.
- Bbox ổn định (jitter thấp).
- Phân loại nhất quán là subtitle.
- Không watermark/scene/person hard risk.
- Geometry hợp lệ.

### Bbox cuối

Dùng **median** hoặc **controlled union**, không lấy bbox từ một frame duy nhất.

### Quy tắc safety

**Gemini OCR không được phép ghi đè safety decision của downstream pipeline.**
Interval `sample5 interval_6` bắt buộc giữ nguyên:
```text
SKIPPED
true_person_overlap_risk
```

---

## 8. Tách OCR, ASR và translation

### Quyết định kiến trúc

```
GeminiVisionOCRProvider      — src/infrastructure/providers/ocr/
GeminiAudioSpeechProvider     — src/infrastructure/providers/speech/
GeminiTranslationProvider     — src/infrastructure/providers/translation/ (giữ nguyên)
```

**Không xây provider kiểu `GeminiVideoUnderstandingProvider`** gộp OCR + ASR + translate.
Lý do:
- **Provenance rõ**: biết chính xác output đến từ model nào, prompt nào.
- **Retry riêng**: lỗi OCR không kéo theo retry ASR/translation.
- **Cache riêng**: cache key khác nhau cho image vs audio vs text.
- **Benchmark riêng**: đo CER cho ASR, đo character accuracy cho OCR.
- **Thay model riêng**: OCR dùng Vision model, ASR dùng Audio model.
- **Phân biệt lỗi**: lỗi OCR khác với lỗi ASR hay translation.
- **Tránh ASR text bị map vào OCR bbox** nếu gộp chung.
- **Giữ Application services độc lập**: timeline mapping, merge policy, safety decision.

### Provenance metadata bắt buộc

Mỗi output phải lưu:

```json
{
  "provider": "gemini",
  "model": "gemini-2.0-flash-lite",
  "prompt_version": "gemini_ocr_v1",
  "schema_version": "ocr_detection_v1",
  "source_kind": "image",
  "media_sha256": "abc123...",
  "retry_count": 0,
  "input_tokens": 1423,
  "output_tokens": 89
}
```

---

## 9. Model routing

### Nguyên tắc

- **Không khóa cứng tên model** trong code.
- Model nằm trong config, cho phép override qua environment variables.

### Cấu hình đề xuất

```yaml
providers:
  ocr:
    gemini_vision:
      model: ${GEMINI_OCR_MODEL:-gemini-2.0-flash-lite}
      escalation_model: ${GEMINI_OCR_ESCALATION_MODEL:-gemini-2.5-flash}

  speech:
    gemini_audio:
      model: ${GEMINI_ASR_MODEL:-gemini-2.0-flash-lite}
      escalation_model: ${GEMINI_ASR_ESCALATION_MODEL:-gemini-2.5-flash}

  translation:
    gemini:
      model: ${GEMINI_TRANSLATION_MODEL:-gemini-2.0-flash-lite}
```

### Chiến lược

```text
Flash-Lite hoặc model rẻ làm pass chính
→ validation pass: accept
→ ambiguous: retry bằng model mạnh hơn (escalation_model)
→ vẫn khác biệt: REVIEW_REQUIRED
```

- Không tự động retry mọi request bằng model mạnh.
- Không dùng model preview làm mặc định nếu có stable model phù hợp.

---

## 10. Cache

### Cache key

```text
SHA256(
  media bytes
  + model name
  + prompt version
  + schema version
  + generation config
)
```

### Cache riêng cho từng loại

| Loại | Cache scope |
|---|---|
| OCR full frame | Image bytes + model + prompt + schema |
| OCR crop retry | Crop image bytes + model + prompt + schema |
| ASR audio | Audio bytes + model + prompt + schema |
| Translation batch | Text input + model + prompt + schema |

**Không gọi lại API nếu input và contract không đổi.**
Cache phải lưu provenance và usage metadata.
**Không log API key trong cache.**

---

## 11. Error handling

### Mapping lỗi

| Lỗi Gemini | ProviderError type | Ghi chú |
|---|---|---|
| 429 | `ProviderRateLimitError` | Retry với backoff |
| Timeout | `ProviderTimeoutError` | Retry |
| Invalid JSON | `ProviderResponseError` | Retry 1 lần với prompt chặt hơn |
| Schema mismatch | `ProviderResponseError` | Retry 1 lần |
| Empty response | `ProviderResponseError` | Retry 1 lần |
| Blocked response | `ProviderBlockedError` | Fail closed |
| Upload failure | `ProviderIOError` | Retry |
| Invalid timestamps | `ProviderValidationError` | Fail closed |
| Invalid bbox | `ProviderValidationError` | Fail closed |
| Model unavailable | `ProviderUnavailableError` | Retry với escalation model |

### Quy tắc xử lý

```text
retryable network/rate error → retry có backoff (max_retries: 5)
invalid model output → tối đa 1 retry với prompt/schema chặt hơn
vẫn invalid → fail closed hoặc REVIEW_REQUIRED
```

**Không tự chế transcript hoặc OCR text.**
**Không tạo placeholder.**

---

## 12. Bảo mật và Free Tier

### Quy tắc

| Rule | Mô tả |
|---|---|
| API key | Chỉ lấy từ `$env:GEMINI_API_KEY` hoặc secret store. **Không lưu trong config repository.** |
| Logging | **Không log raw authorization headers** hoặc API keys. |
| Free Tier data policy | Không gửi dữ liệu nhạy cảm nếu chưa chấp thuận chính sách dữ liệu Google. |
| Cleanup | Xóa uploaded files trên Google servers sau khi job hoàn thành (nếu API hỗ trợ). |
| Artifact storage | **Không lưu raw media response** ngoài artifact policy đã định nghĩa. |

### Free Tier status

**Free Tier phù hợp cho development/qualification trước.**
**Không tự động coi là production-safe.**
Cần review data processing agreement trước khi dùng Free Tier cho dữ liệu sản xuất.

---

## 13. Lộ trình triển khai

### Phase A — Documentation và contracts (hiện tại)

- [x] Hoàn thành tài liệu này.
- [ ] Khóa config và provenance contract.
- Chưa đổi code runtime.

### Phase B — Shared Gemini infrastructure

Tạo tại `src/infrastructure/gemini/`:

```text
src/infrastructure/gemini/
├── __init__.py
├── gemini_client.py          # Gemini API client wrapper (lazy init, API key mgmt)
├── request_scheduler.py      # Shared RPM/TPM scheduler với token tracking
├── response_cache.py         # Disk-based response cache
└── file_manager.py           # Upload/cleanup temporary files
```

### Phase C — Gemini OCR provider

```text
src/infrastructure/providers/ocr/gemini_vision_ocr_provider.py
```

- Implement `OCRProvider` port.
- Chạy shadow mode đầu tiên (ghi log so sánh với PaddleOCR, không ảnh hưởng output).
- Chưa đổi default provider.

### Phase D — Gemini ASR provider

```text
src/infrastructure/providers/speech/gemini_audio_speech_provider.py
```

- Implement `SpeechProvider` port.
- Mặc định full-audio request cho video dưới 5 phút.
- Chạy shadow mode.

### Phase E — Benchmark

```text
OCR:
PaddleOCR vs Gemini full-frame vs Gemini full-frame + crop retry

ASR:
Faster-Whisper small vs Gemini full-audio
```

- Metrics: accuracy, runtime, token usage, cost, retry rate.
- Báo cáo benchmark tại `docs/GEMINI_BENCHMARK_REPORT.md`.

### Phase F — Hybrid escalation

- Chỉ retry các frame/audio không đạt validation.
- Dùng escalation model khi pass đầu không đủ confidence.

### Phase G — Chuyển default

- Đổi default provider sau khi benchmark và visual QA đạt acceptance.
- **Không xóa PaddleOCR/Faster-Whisper.** Giữ lại trong codebase làm fallback.

---

## 14. Acceptance criteria

### OCR

| Tiêu chí | Mô tả |
|---|---|
| Subtitle detection recall | Không giảm so với PaddleOCR baseline |
| OCR text quality | Được human review |
| Source text hallucination | Zero tolerance — không tự thêm chữ |
| Bbox jitter | Trong ngưỡng cho phép |
| Watermark/scene text false cover | Zero |
| Dynamic replacement safety | Không giảm — `sample5` hard reject giữ nguyên |

### ASR

| Tiêu chí | Mô tả |
|---|---|
| Reference transcript | Phải có reference transcript để đo |
| CER/WER | Đo và báo cáo |
| Timestamp deviation | Độ lệch trung bình < 0.5s |
| End-of-audio text loss | Zero |
| Duplicate segments | Zero |
| Summary/hallucination | Zero |
| Proper-name và homophone errors | Được audit |

### End-to-end

| Metric | Target |
|---|---|
| `placeholder_count` | 0 |
| `source_copy_without_reason` | 0 |
| `untranslated_cjk_in_vietnamese` | 0 |
| Provenance | OCR, ASR, translation đầy đủ |
| Hard safety reject | Giữ nguyên |
| Source leakage review | Được thực hiện |
| `production_ready` | **NO** (cho đến khi qualification hoàn tất) |

---

## 15. Kết luận chiến lược

> Với video dưới 5 phút, chiến lược mặc định là **một Gemini ASR request cho toàn bộ audio**, **một hoặc vài Gemini OCR batch request trên full frames**, **crop retry chỉ cho trường hợp mơ hồ**, và **một Gemini translation batch request**. OCR không cần OCR local để tạo crop; crop được xác định từ bbox Gemini và chỉ là thao tác xử lý pixel cục bộ. RPM và concurrency phải cấu hình được và dùng chung một scheduler, không hardcode quota thật của Google.

### Các quyết định kiến trúc đã khóa

1. Ba provider riêng — không gộp.
2. Full-audio ASR single request — không chunk mặc định.
3. Full-frame OCR pass đầu — crop retry chỉ khi mơ hồ.
4. Temporal consensus từ evidence — không tin LLM confidence.
5. Shared request scheduler — không limiter riêng.
6. Model routing theo config — không hardcode.
7. Fail-closed safety — Gemini không ghi đè downstream safety.
8. PaddleOCR và Faster-Whisper giữ nguyên trong codebase.

---

*Tài liệu tham khảo: `PROJECT_OPERATING_SPECIFICATION.md`, `FRAMEWORK_V1_SIGNOFF.md`, `ROADMAP_AFTER_V1.md`, `ASR_BENCHMARK_REPORT.md`, `REAL_OUTPUT_QUALITY_REPORT.md`, `static_source_cover_mvp_baseline.md`, `static_source_cover_mvp_v2_visual_qa.md`, `phase9_v5_replay_audit.md`, `phase9b_same_source_baseline.md`, `phase9g_local_safety_audit.md`, `project_reassessment_phase8.md`, `resource_policy.md`*
