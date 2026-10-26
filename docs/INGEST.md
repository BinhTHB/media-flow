# Ingest (Source Download)

Bước đầu của chuỗi reup: URL → file local mà pipeline xử lý được.

```
URL (YouTube / TikTok / Douyin / Bilibili / Instagram / ...)
  → yt-dlp
  → input/downloads/<id>.mp4
  → manifest.json (idempotent)
  → [tuỳ chọn] pipeline
```

## Cách dùng

```bash
# Một URL
python -m src.interfaces.ingest --url "https://www.youtube.com/watch?v=xxx"

# Nhiều URL từ file (mỗi dòng một URL, '#' là comment)
python -m src.interfaces.ingest --urls-file links.txt --out input/downloads

# Chỉ lấy 30 giây đầu, chất lượng tối đa 720p
python -m src.interfaces.ingest --url "..." --quality 720 --trim 30

# Tải xong chạy luôn pipeline (dịch phụ đề + lồng tiếng)
python -m src.interfaces.ingest --url "..." --then-process \
    --subtitle-action translate --dub-audio

# Nguồn cần đăng nhập
python -m src.interfaces.ingest --url "..." --cookies-from-browser chrome
# hoặc
python -m src.interfaces.ingest --url "..." --cookie-file secrets/cookies.txt
```

## Flags

| Flag | Chức năng |
|---|---|
| `--url` | URL nguồn (lặp lại được) |
| `--urls-file` | File danh sách URL |
| `--out` | Thư mục tải (mặc định `input/downloads`) |
| `--quality` | `best` \| `1080` (mặc định) \| `720` \| `480` |
| `--trim` | Chỉ giữ N giây đầu (0 = cả video) |
| `--subtitles` | Tải phụ đề nếu nguồn có (rẻ hơn OCR) |
| `--overwrite` | Tải lại nguồn đã có trong manifest |
| `--cookie-file` / `--cookies-from-browser` | Nguồn cần auth |
| `--proxy` | Proxy cho downloader |
| `--json` | In kết quả dạng JSON |
| `--then-process` | Chạy pipeline sau khi tải |

## Idempotency

`manifest.json` trong thư mục tải ghi mỗi URL đã xử lý. Chạy lại cùng URL →
`SKIPPED`, không tải lại. Nếu file trên đĩa biến mất, lần sau sẽ tải lại (không
báo success giả).

`--overwrite` bỏ qua manifest.

## Retry

| Loại lỗi | Hành vi |
|---|---|
| Timeout, connection reset, HTTP 429/5xx | `TransientIngestError` → backoff 5s → 10s → 20s |
| Private, unavailable, geo-block, age-restricted, unsupported URL | Permanent, không retry |
| Lỗi không phân loại được | Coi là permanent (retry chỉ nhân thời gian chờ) |

## Chất lượng & format

- `--quality` đặt **height cap**, không ép đúng độ phân giải. `bv*[height<=N]+ba`
  chọn video tốt nhất dưới ngưỡng + audio tốt nhất.
- Output ưu tiên mp4 (`merge_output_format`), để các stage sau không phải remux.
- `--trim` dùng `download_ranges` của yt-dlp: cắt ngay lúc tải, không tải cả
  video 40 phút để lấy 30 giây.

## Kiến trúc

```
src/core/models/ingest.py            # SourceSpec, IngestResult, IngestBatch, infer_platform
src/core/contracts.py                # DownloaderPort, TransientIngestError
src/features/ingest/service.py       # retry/backoff, manifest idempotency
src/infrastructure/ingest/yt_dlp.py  # yt-dlp adapter (lazy import)
src/interfaces/ingest/               # CLI + --then-process chaining
```

Thêm downloader khác (ví dụ API riêng của Douyin): implement `DownloaderPort`,
không sửa `IngestService`.

## Lưu ý

- Chỉ tải nội dung bạn có quyền sử dụng.
- yt-dlp là optional dependency nhưng được cài trong `requirements.txt` vì
  ingest là bước mặc định của chuỗi reup.
- File tải về nằm trong `input/` — đã gitignore.

## Test

```bash
python -m pytest tests/test_ingest.py -q
```

Test dùng fake downloader, không chạm mạng. Adapter yt-dlp chỉ được test qua
các hàm thuần (format selector, error mapping, resolve output).
