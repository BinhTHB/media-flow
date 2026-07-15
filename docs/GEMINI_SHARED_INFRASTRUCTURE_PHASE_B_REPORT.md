# Báo cáo kết quả triển khai Phase B — Shared Gemini Infrastructure

> **Dự án**: AI Video Translator (`media-flow`)
> **Tài liệu**: `docs/GEMINI_SHARED_INFRASTRUCTURE_PHASE_B_REPORT.md`
> **Trạng thái**: IMPLEMENTED / QUALIFICATION INFRASTRUCTURE ONLY
> **Production Ready**: NO

---

## 1. Scope (Phạm vi)
Tài liệu này báo cáo kết quả thiết kế và triển khai Phase B, xây dựng lớp hạ tầng Gemini dùng chung cho toàn bộ dự án (`media-flow`), đồng thời tái cấu trúc (refactor) tích hợp ngược với `GeminiTranslationProvider` để kiểm chứng.

Phạm vi Phase B loại trừ hoàn toàn việc tạo mới OCR provider, ASR provider, chạy benchmark hoặc gọi API thực tế của Google.

## 2. SDK được chọn
- **Quyết định**: Sử dụng giao thức REST thuần dựa trên thư viện chuẩn `urllib.request` của Python thay vì cài đặt các gói SDK ngoài như `google-genai` hoặc `google-generativeai`.
- **Lý do**:
  - Không phát sinh thêm dependency mới trong dự án.
  - Tương thích hoàn toàn với môi trường offline và mock tests.
  - Khởi tạo lazy (lazy initialization) hoàn toàn độc lập, không gây lỗi khi chạy các CLI mock/local nếu thiếu optional libraries.

## 3. Module Map
Gói hạ tầng dùng chung được đặt tập trung trong `src/infrastructure/gemini/`:

- `__init__.py`: Cung cấp public exports.
- `contracts.py`: Chứa các dataclass trao đổi thông tin (`GeminiRequest`, `GeminiResponse`, `GeminiUsage`, `GeminiModelInfo`).
- `exceptions.py`: Định nghĩa các lỗi chi tiết (`ProviderRateLimitError`, `ProviderTimeoutError`,...) kế thừa từ `ProviderError`.
- `client.py`: Wrapper client đồng bộ, thực hiện REST call với lazy key resolution.
- `runtime.py`: `GeminiRuntime` là container quản lý phạm vi ứng dụng (application-scoped) sở hữu các shared instances.
- `request_scheduler.py`: Bộ điều phối RPM/TPM/concurrency dùng chung.
- `response_cache.py`: Cơ chế lưu đĩa JSON cache với atomic write.
- `structured_output.py`: Bộ bóc tách và phân tích cú pháp structured JSON.
- `model_validator.py`: Xác thực tính khả dụng và lifecycle của model.
- `file_manager.py`: Bộ quản lý tải tệp tin no-op/mock-safe.
- `usage_recorder.py`: Lưu trữ metrics sử dụng token/cache mà không rò rỉ secret.

## 4. Dependency Ownership
- **Giao diện công khai (Ports)**: Giữ nguyên `OCRProvider`, `SpeechProvider`, và `TranslationProvider` thuộc lớp Application.
- **Phân tách ranh giới (Boundaries)**: Lớp Domain/Application hoàn toàn không biết đến Gemini SDK hoặc các tệp tin trong `src.infrastructure.gemini`.
- **Composition Root**: `ApplicationBuilder` chịu trách nhiệm khởi tạo `GeminiRuntime` duy nhất từ cấu hình và truyền (inject) vào `ProviderFactory`.

## 5. Config và Environment Overrides
Cấu hình Gemini được định nghĩa dạng typed dataclass `GeminiConfig` trong `AppConfig` với các trường:
- `model`: Định dạng model chính (default: `gemini-3.1-flash-lite`).
- `max_retries`: Số lần thử lại (default: 3).
- `cache_dir`: Đường dẫn thư mục cache.
- `cache_namespace`: Namespace riêng của cache (default: `shared`).
- `cache_enabled`: Bật/tắt cache (default: true).

Hỗ trợ các ghi đè biến môi trường tường minh với validation (raise `ValueError` -> `ValidationError` nếu sai định dạng số/boolean):
- `GEMINI_MODEL`
- `GEMINI_MAX_RETRIES`
- `GEMINI_CACHE_DIR`
- `GEMINI_CACHE_NAMESPACE`
- `GEMINI_CACHE_ENABLED`
- `GEMINI_FILE_MANAGER_ENABLED`

*Lưu ý: API Key (`GEMINI_API_KEY`) được giữ hoàn toàn ở môi trường bên ngoài hoặc nạp lazy trực tiếp ở Runtime, không được lưu trữ hay serialize trong `AppConfig`.*

## 6. Scheduler Algorithm
- Sử dụng thuật toán rolling window dựa trên `time.monotonic()` để theo dõi RPM và TPM chính xác, thay vì dùng `sleep(60)` cố định.
- Hỗ trợ cơ chế semaphore thread-safe để kiểm soát concurrency (mặc định: `max_concurrency: 1`).
- Đảm bảo giải phóng semaphore (`release()`) trong khối `finally` khi có lỗi xảy ra.

## 7. Retry Policy
- Chỉ thực hiện retry tự động trên lỗi kỹ thuật (transient errors như HTTP 429 rate limit, 5xx server error, timeout).
- Sử dụng exponential backoff có giới hạn tối đa (`backoff_max_seconds`) và được tính toán kèm bounded random jitter.
- Tôn trọng giá trị header `Retry-After` (nếu có) khi được bật trong cấu hình.
- Các lỗi về nghiệp vụ (invalid schema, safety block, missing files) lập tức dừng và raise lỗi, không thử lại.

## 8. Cache Key
Khóa cache được mã hóa SHA256 dựa trên cấu trúc canonical JSON sắp xếp khóa (sort_keys) gồm:
- `input_sha256`: Hash của nội dung đầu vào.
- `model`: Model ID chính xác.
- `model_role`: Vai trò model (`primary`/`escalation`).
- `prompt_version`: Phiên bản prompt.
- `schema_version`: Phiên bản schema.
- `generation_config`: Cấu hình nhiệt độ/tham số.
- `media_resolution`: Độ phân giải nếu có.
- `api_surface`: Tên API được gọi (mặc định: `"generate_content"`).

## 9. Cache Storage Layout
- Cache lưu trữ dưới dạng các tệp JSON riêng biệt phân cấp theo namespace trên đĩa đệm.
- Quá trình ghi cache được thực hiện nguyên tử (atomic write): Ghi ra file tạm rồi ghi đè (replace) để tránh hỏng dữ liệu khi nhiều tiến trình chạy song song.
- Tách biệt vùng cache (namespace) giữa `primary` và `escalation` model, đảm bảo không có sự dùng chéo cache.

## 10. Model Validation
- Sử dụng API `models.get` và `models.list` ở runtime để kiểm tra tính khả dụng của model được cấu hình trước khi gửi request thực tế đầu tiên.
- Quan sát trường `modelStatus.modelStage` và `retirementTime` trả về để chặn các model `LEGACY`, `DEPRECATED`, `RETIRED` và cảnh báo nếu sử dụng `PREVIEW`.

## 11. File Lifecycle
- `GeminiFileManager` cung cấp cơ chế đăng ký và dọn dẹp (cleanup) các tệp tin tạm được upload phục vụ multimodal input.
- Tự động cleanup ngay sau khi request/job kết thúc kể cả khi có ngoại lệ xảy ra.

## 12. Usage/Provenance Schema
- `UsageRecorder` ghi lại đầy đủ thông tin: tokens (prompt/candidates), cache hit, retry count, model ID, API surface và thời điểm thực hiện.
- Mọi bản ghi usage và provenance đều bị loại bỏ/lọc bỏ API key hoặc nội dung prompt thô.

## 13. Tích hợp GeminiTranslationProvider
- `GeminiTranslationProvider` được tái cấu trúc để nhận shared `GeminiRuntime` được inject từ `ProviderFactory`.
- Nhà cung cấp dịch vụ dịch thuật giờ đây ủy thác (delegate) việc gọi API, retry, rate limiting, cache và recording hoàn toàn cho lớp hạ tầng dùng chung.
- Giữ nguyên toàn bộ hành vi dịch thuật cũ: batching 15 segments/request, phân tích numbered lines, và bảo toàn kết quả đầu ra trong tests.

## 14. Tests
- 100% các test của Phase B chạy ở chế độ offline (mock urllib.urlopen) và không yêu cầu API key thật.
- Bổ sung kiểm thử composition (`tests/test_composition.py`) chứng minh một `GeminiRuntime` duy nhất được chia sẻ và inject vào các Gemini providers thuộc cùng một application context.
- Bài test lỗi API key và parsing được bảo toàn hoàn hảo.

## 15. Known Limitations (Hạn chế đã biết)
- **Single Process**: Scheduler hiện tại được thiết kế chạy đơn tiến trình (single process/in-memory). Hạn ngạch đa tiến trình (multi-process distributed scheduling) nằm ngoài phạm vi thiết kế Phase B.
- **Standard API**: Hiện tại chỉ hỗ trợ `generate_content` (Standard API). Interactions/Streaming API chưa được triển khai.

## 16. Handoff cho Phase C
Lớp hạ tầng `src/infrastructure/gemini/` đã sẵn sàng 100% để Phase C (`GeminiVisionOCRProvider`) và Phase D (`GeminiAudioSpeechProvider`) kế thừa trực tiếp:
- Phase C chỉ cần lấy `GeminiClient` từ runtime qua: `runtime.get_client(role="primary")`.
- Sử dụng `request_scheduler` và `response_cache` của runtime để tự động ăn khớp với translation quota.

## 17. Production Ready: NO
Hạ tầng mới chỉ hoàn thành việc tích hợp dịch thuật và cấu trúc dùng chung. Hệ thống chưa sẵn sàng cho môi trường production.
