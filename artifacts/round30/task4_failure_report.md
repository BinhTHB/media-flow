# Báo cáo lỗi E2E — Round 30 / TASK 4

## Kết luận

**FAIL.** Theo quy tắc chặn của vòng kiểm thử, không thể xác nhận PASS và không được chuyển sang đóng gói.

## Lệnh đã chạy

Lệnh E2E đã được chạy với chế độ subtitle placement và các cờ V2.1 bắt buộc cho job `round30-sample5`. Log E2E chỉ lưu phần thực thi sau đây (không chứa bí mật):

```text
.\.venv\Scripts\python.exe src\interfaces\cli\demo.py ...
```

Các cờ subtitle placement và V2.1 đã được áp dụng trong lần chạy này. Chuỗi đối số đầy đủ không được ghi lại trong `e2e_round30.log`; vì vậy báo cáo này không suy diễn hoặc bịa thêm đối số không có trong bằng chứng.

## Môi trường GPU

- GPU: `NVIDIA GeForce RTX 4050`
- Driver NVIDIA: `581.29`
- CUDA: `13.0`
- GPU khả dụng trong lúc kiểm tra (WDDM; `154 MiB / 6141 MiB` được sử dụng).

## Kết quả thực thi E2E

- Exit code: **1**
- Lỗi chấm dứt:

```text
Failed to generate subtitle placement debug artifacts:
Cannot generate after_render.jpg because rendered output.mp4 is missing.
```

Do đó lần chạy E2E không hoàn tất phần tạo debug artifact subtitle placement.

## Kết quả self-check bắt buộc

Nguồn: `artifacts/round30/e2e_self_check.json`

| Trường | Giá trị |
|---|---:|
| `timestamp_utc` | `2026-07-27T05:17:06.465084+00:00` |
| `video` | `E:\projects\media-flow\output\round30-sample5\final.mp4` |
| `ass` | `E:\projects\media-flow\output\round30-sample5\round30-sample5\subtitles\subtitle.ass` |
| `ffprobe_stream.codec_name` | `h264` |
| `ffprobe_stream.width` | `720` |
| `ffprobe_stream.height` | `1280` |
| `ffprobe_stream.pix_fmt` | `yuv420p` |
| `ass_has_t003_provenance` | `true` |
| `ass_mask_count` | `235` |
| `h264_720x1280` | `true` |
| `t003_frame_read` | `true` |
| `t003_dark_ratio` | `0.03463650173611111` |
| `t003_dark_ratio_high` | `false` |
| `status` | **`FAIL`** |

## Lý do FAIL

Mặc dù tệp ASS có `235` dòng mask, provenance T003 hiện diện và video H.264 có kích thước `720x1280`, self-check bắt buộc trả về **FAIL** vì `t003_dark_ratio_high` là `false` (`t003_dark_ratio = 0.03463650173611111`). Đồng thời, lệnh E2E kết thúc với exit code `1` do thiếu `rendered output.mp4` cần để tạo `after_render.jpg`.

## Lưu ý về phép đo checker

`t003_dark_ratio` hiện được checker tính trên **toàn bộ khung hình**, không phải riêng crop T003. Điều này giải thích vì sao giá trị có thể thấp so với phép đo vùng T003 mong đợi. Tuy nhiên, đây chỉ là lưu ý về phạm vi đo; **không** được dùng nó để suy ra PASS. Checker bắt buộc hiện tại vẫn trả về **FAIL** và quyết định gate giữ nguyên là **FAIL**.

## TASK 5

**TASK 5 đã chủ động không chạy.** Theo quy tắc gate failure, không tạo package và không thực hiện đóng gói.
