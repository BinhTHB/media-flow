# Publishing / Re-upload Automation

Tính năng reup video (video hoàn thiện trong queue → sinh metadata → upload →
ghi state → không bao giờ upload trùng).

Hai họ target, chọn theo nhu cầu:

| Họ | Target | Cơ chế | Phù hợp |
|---|---|---|---|
| **Official API** | `youtube` | OAuth refresh token + Data API v3 resumable upload | Kênh của chính bạn, cần bền, đúng ToS |
| **Browser session** | `browser:tiktok:<account>` | Playwright + cookie jar đã đăng nhập | Nền tảng không cho API reup, cần đa tài khoản |
| **Local** | `file:DIR` | Copy + sidecar | Dry-run, hand-off cho tool khác |

## Giới hạn thật của từng nền tảng (10/2026)

### YouTube Data API v3

- **`videos.insert` có bucket riêng: 100 calls/ngày**, mỗi call 1 quota.
  (Không còn tính 1.600 units như tài liệu cũ.) 10.000 units/ngày còn lại
  dành cho các endpoint khác.
- **Service account KHÔNG upload được** — bắt buộc OAuth refresh token của
  user thật.
- Quota reset lúc nửa đêm giờ Pacific.

### TikTok Content Posting API

Đọc kỹ trước khi chọn hướng API:

- Chưa audit app: chỉ post được `SELF_ONLY` (private), cap **5 user/24h**.
- Đã audit: cap **~15 post/ngày/tài khoản creator**, dùng chung giữa mọi client.
- Guidelines ghi rõ **không chấp nhận** "an app that copies arbitrary contents
  from other platforms to TikTok" — tức reup thuần qua API là vi phạm.
- Cấm thêm watermark/branding của tool lên video.

Vì vậy reup đa nền tảng dùng **browser session**, không dùng Content Posting API.

## Re-upload variation

Nền tảng fingerprint media; file giống hệt bản gốc bị match ngay. Mỗi lần
publish, worker render một **variant** khác bản master:

| Knob | Khoảng | Ghi chú |
|---|---|---|
| `speed` | 0.90–1.10 | video `setpts` + audio `atempo` |
| `crop_ratio` | 0–0.10 | crop rồi scale về đúng resolution gốc |
| `mirror` | bool | `hflip` |
| `rotate_degrees` | ±3° | xoay + crop góc đen |
| `pitch_semitones` | ±2 | `asetrate` + `atempo` bù |
| `brightness/contrast/saturation` | nhỏ | `eq` filter |

Mức: `--variation off|light|balanced|aggressive` (mặc định `balanced`).

Job xử lý ghi biến thể này vào chính file output. Cổng chất lượng đọc `variation` trong `job.json` và chấm file đó. Sidecar `<tên>.variation.json` nằm cạnh file. Worker thấy sidecar thì đăng nguyên file, không biến đổi lần hai. File bỏ thẳng vào hàng chờ, không có sidecar, vẫn được worker biến đổi lúc đăng.

**Deterministic**: profile = f(item_id, attempt). Retry cùng attempt cho ra
đúng cùng bytes; attempt mới cho variant mới. Master file không bao giờ bị sửa.

Variant là artifact tạm — xóa ngay sau khi upload xong.

## Zernio

Studio tab **Đăng Zernio** lưu mỗi nền tảng một API key tại `secrets/zernio_accounts.json`. Key không hiện đầy đủ trên màn hình. Bấm kiểm tra gọi `GET /v1/accounts` của key đó. Đăng một file đã xử lý thì mỗi key tải video lên `POST /v1/media/presign` rồi `POST /v1/posts` với `publishNow` tới tài khoản cùng nền tảng.

Hai tài khoản kết nối đầu của một team Zernio là miễn phí. Key thứ ba trở đi do Zernio tính phí.

## Cách dùng

```bash
# 1. Nạp video vào queue (sidecar metadata tùy chọn)
cp outputs/jobs/job_xxx/output.mp4 outputs/publish_queue/ep01.mp4
# ep01.publish.json -> {"title": "...", "description": "...", "tags": [...], "privacy": "public"}

# 2. Official API (YouTube)
python -m src.interfaces.publish_worker \
    --queue outputs/publish_queue --target youtube --variation balanced

# 3. Browser session (TikTok), đa tài khoản
python -m src.interfaces.publish_login --platform tiktok --account acc1
python -m src.interfaces.publish_worker \
    --queue outputs/publish_queue --target browser:tiktok:acc1 --watch --interval 60

# 4. Nhiều target cùng lúc
python -m src.interfaces.publish_worker \
    --queue outputs/publish_queue \
    --target file:E:/reup/delivered \
    --target youtube \
    --target browser:tiktok:acc1
```

Metadata: sidecar JSON → Gemini (nếu có `GEMINI_API_KEY`, trừ khi
`--no-metadata`) → fallback tên file. Mọi lỗi metadata đều fallback; worker
không chết vì lý do này.

State: `outputs/publishing/state.json` — mỗi (video, platform) publish một lần.

## Browser session

### Cài đặt

```bash
pip install playwright
playwright install chromium
```

Playwright là optional: thiếu nó thì chỉ target browser báo lỗi cấu hình rõ
ràng, phần còn lại của MediaFlow vẫn chạy.

### Tạo session (một lần / tài khoản)

```bash
python -m src.interfaces.publish_login --platform tiktok --account acc1
```

Mở browser thật, bạn đăng nhập (password + 2FA + captcha nếu có), helper poll
đến khi hết login wall rồi lưu cookie jar. Timeout mặc định 300s.

Session lưu tại `outputs/publishing/sessions/<platform>_<account>.json`
(gitignored, quyền 0600). Đây là credential — không log, không commit.

### Session hết hạn

Publisher phát hiện login wall → `invalidate()` session → raise
`TransientPublishError` với hướng dẫn chạy lại login helper. Không retry vô
hạn trên cookie chết.

## Kiến trúc

```
src/core/models/variation.py          # VariationProfile (bounded, serializable)
src/core/models/account.py            # AccountIdentity, PostingWindow, SessionState
src/core/models/publication.py        # PublishItem, VideoMetadata, PublishRecord
src/core/contracts.py                 # PublicationPort, VideoVariationPort, SessionStorePort
src/features/publishing/
    service.py                        # retry/backoff, idempotency, variation wiring
    variation.py                      # VariationPolicy (deterministic seeding)
src/infrastructure/publishing/
    youtube.py                        # Data API v3 resumable upload
    browser_base.py                   # BrowserPublisher template method
    tiktok_browser.py                 # TikTok upload page flow
    session_store.py                  # cookie jar per account
    publication_store.py              # JSON state
    file_target.py                    # local delivery
    gemini_metadata.py                # AI metadata
src/infrastructure/video/variation.py # FFmpeg filter chain
src/interfaces/publish_worker/        # CLI/daemon
src/interfaces/publish_login/         # one-time session creation
```

Thêm nền tảng mới:
- API → implement `PublicationPort`.
- Browser → subclass `BrowserPublisher`, override `_open_upload_page` /
  `_submit`, thêm một dòng vào `build_browser_publisher`.

Không sửa use case.

## Lưu ý tuân thủ

- Chỉ reup nội dung bạn có quyền.
- Session-based automation vi phạm ToS của hầu hết nền tảng. Rủi ro khóa tài
  khoản thuộc về người vận hành.
- Đừng dùng tool để né phát hiện bản quyền hoặc spam.

## Test

```bash
python -m pytest tests/test_publishing.py tests/test_publishing_session.py \
    tests/test_variation.py -q

# Integration thật với ffmpeg (opt-in)
$env:MEDIA_FLOW_RUN_MEDIA_INTEGRATION="1"
python -m pytest tests/test_variation_ffmpeg.py -q
```
