# Publishing / Re-upload Automation

Tính năng reup video tự động (no-human-in-loop): video hoàn thiện trong queue
→ sinh metadata (AI) → upload qua **API chính thức** của nền tảng → ghi state
→ không bao giờ upload trùng.

## Nghiên cứu tổng quan (10/2026)

| Hướng | Đại diện | Đánh giá cho use case này |
|---|---|---|
| SaaS cross-post | Repurpose.io, Blotato, Upload&Post, Opus.pro | Nhanh nhưng thuê bao hàng tháng, dữ liệu qua cloud bên thứ 3, giới hạn target, phụ thuộc vendor |
| Open-source uploader | youtubeuploader (Java, cookie) | Brittle: dựa cookie browser, dễ vỡ khi YouTube đổi, rủi ro tài khoản |
| **API chính thức (chọn)** | YouTube Data API v3, TikTok Content Posting API | Bền, hợp ToS, quota rõ ràng, refresh token cho automation |

Sự kiện then chốt thiết kế:
- **YouTube không chấp nhận service account** cho upload — phải dùng OAuth
  refresh token (lấy 1 lần, dùng mãi mãi cho đến khi bị thu hồi).
- `videos.insert` tốn ~1.600 quota units; mặc định 10.000 units/ngày ≈ **6
  upload/ngày** (xin tăng quota được). Còn ngưỡng ẩn ~1.600 upload/ngày.
- HTTP 429 / 5xx / quotaExceeded là tạm thời → worker backoff lũy tiến
  (30s → 60s → 120s → 240s → 480s), tối đa 5 lần.

## Cách dùng

```bash
# 1. Nạp video vào queue (kèm sidecar metadata tùy chọn)
cp outputs/jobs/job_xxx/output.mp4 outputs/publish_queue/ep01.mp4
# tùy chọn: ep01.publish.json -> {"title": "...", "description": "...", "tags": [...], "privacy": "public"}

# 2. Chạy one-shot
python -m src.interfaces.publish_worker \
    --queue outputs/publish_queue \
    --target file:E:/reup/delivered \
    --target youtube

# 3. Hoặc chạy daemon canh queue mãi mãi
python -m src.interfaces.publish_worker --queue outputs/publish_queue \
    --target youtube --watch --interval 60
```

Metadata: ưu tiên sidecar JSON → nếu có `GEMINI_API_KEY` (và không truyền
`--no-metadata`) thì Gemini sinh title/description/tags tiếng Việt từ tên
file → cuối cùng fallback tên file. Mọi lỗi metadata đều fallback, worker
không bao giờ chết vì lỗi này.

State: `outputs/publishing/state.json` — mỗi (video, platform) chỉ được
publish một lần; restart worker an toàn.

## Thiết lập YouTube (một lần duy nhất)

1. Google Cloud Console → tạo project → bật **YouTube Data API v3**.
2. OAuth consent screen → thêm chính bạn làm test user (app ở chế độ
   "testing" vẫn upload được với refresh token của test user).
3. Tạo OAuth Client (Desktop app) → có `client_id` / `client_secret`.
4. Lấy refresh token với scope `https://www.googleapis.com/auth/youtube.upload`
   (mở URL consent trên máy, đồng ý 1 lần, đổi authorization code lấy
   refresh token).
5. Set env:

```bash
YOUTUBE_CLIENT_ID=...
YOUTUBE_CLIENT_SECRET=...
YOUTUBE_REFRESH_TOKEN=...
```

Privacy mặc định là **private** — muốn public phải ghi rõ trong sidecar
(`"privacy": "public"`) hoặc chỉnh `VideoMetadata.privacy` mặc định. Đây là
chốt an toàn cho automation.

## Kiến trúc (clean architecture)

```
src/core/models/publication.py        # entity: PublishItem, VideoMetadata, PublishRecord
src/core/contracts.py                 # port: PublicationPort, MetadataGenerationPort, TransientPublishError
src/features/publishing/service.py    # use case: retry/backoff, idempotency, queue scan
src/infrastructure/publishing/
    publication_store.py              # JSON state store
    file_target.py                    # port impl: copy tới thư mục đích
    youtube.py                        # port impl: Data API v3 resumable upload
    gemini_metadata.py                # port impl: AI metadata (fallback an toàn)
src/interfaces/publish_worker/        # wiring + CLI/daemon entry
```

Thêm nền tảng mới (TikTok, Facebook) = thêm 1 provider implement
`PublicationPort`; không sửa use case.

## Lưu ý tuân thủ

- Dùng API chính thức với tài khoản/OAuth của chính bạn — đúng ToS nền tảng.
- Chỉ reup nội dung bạn có quyền (video do media-flow dịch/lồng tiếng từ
  nguồn được phép). Trách nhiệm nội dung thuộc về người vận hành.
- Không dùng tool để né phát hiện bản quyền hoặc spam; quota API tự giới hạn.

## Test

```bash
python -m pytest tests/test_publishing.py -q   # 14 test, mock/fake session, 0 network
```
