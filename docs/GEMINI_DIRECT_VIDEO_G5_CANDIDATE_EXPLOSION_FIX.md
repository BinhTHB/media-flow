# Gemini Direct-Video OCR candidate explosion fix (Phase G.5)

## 1. Bối cảnh
Trong Phase G.5, frame-window OCR được thay thế bằng pipeline Gemini direct-video. Trong quá trình chạy live, một lỗi "candidate explosion" đã xảy ra trên video sample4 (duration ~24s) do:
- V1 Discovery tạo ra quá nhiều candidates trùng lặp (ví dụ: phát hiện theo từng frame thay vì gom nhóm thành temporal track).
- Thiếu logic deduplication deterministic sau stage V1.
- V2 classification được xếp hàng (queue) chạy đồng thời cho toàn bộ candidates trước khi lọc, dẫn tới số lượng request downstream (V3/V4/V5/Translation) tăng bùng nổ, gây cạn kiệt quota API và kéo dài thời gian chạy.

## 2. Thiết kế giải pháp
Để giải quyết triệt để blocker này, các cải tiến kỹ thuật sau đã được triển khai:
- **Deduplication deterministic sau V1**: Gộp các candidate có bounding box tương đồng (IoU > 0.5) và khoảng cách thời gian ngắn (gap < 1.0s) thành các temporal track hợp nhất. Dùng robust median để tìm representative bounding box.
- **Global Event-ID Namespacing**: Đảm bảo event ID của V3 luôn có namespace duy nhất dạng `{candidate_id}:{raw_event_id}` để tránh xung đột dữ liệu (collision) đè lên cấu trúc lưu trữ geometry/transcription/translation ở downstream.
- **V3 Timestamp Disambiguation**: Phân định rõ ràng relative-to-clip và source-absolute timestamps trả về từ Gemini, ưu tiên relative check khi clip_start > 0.
- **Stage Timeout và No-Progress Watchdog**: Áp dụng thời gian timeout tối đa cho mỗi stage request (default = 360s) để tránh treo vô hạn, cùng với watchdog kiểm tra no-progress.
- **Request Planning**: Tính toán minimum/maximum requests trước khi dispatch downstream. Thực hiện fail-fast nếu minimum requests vượt giới hạn cho phép.
- **Checkpointing**: Ghi file checkpoint nguyên tử sau mỗi stage chính (`candidates_raw`, `candidates_deduplicated`, `request_plan`, v.v.) và đảm bảo chúng được lưu trữ khi bị gián đoạn hoặc kết thúc.

## 3. Kiểm thử
Tất cả các trường hợp biên và logic nghiệp vụ đã được bao phủ bởi 29 unit tests chuyên biệt trong `tests/test_gemini_video_direct_ocr_provider.py` (tất cả đều Pass, không có regression trên 507 test suite).

## 4. Trạng thái Production Ready: NO
Hệ thống đã sẵn sàng về mặt cấu trúc và thuật toán để chặn candidate explosion, tuy nhiên do giới hạn quota API Free Tier trên môi trường live hiện tại (gặp lỗi Rate Limit HTTP 429 sau vài requests ban đầu), pipeline chạy live chưa thể xác nhận E2E hoàn thành toàn bộ sample mà không bị nghẽn mạng. Cần nâng cấp quota API hoặc cấu hình retry dài hạn trước khi chuyển trạng thái sang Production Ready.
