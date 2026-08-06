# YOLO Burn-in Detection

Tài liệu này mô tả cách model **YOLO26n fine-tune** (4 lớp) được tích hợp vào
pipeline của AI Video Translator, cách cấu hình, chạy kiểm tra, và các lỗi
thường gặp.

## 1. Model đang dùng

- **File**: `models/yolo/best.pt` (YOLO26n fine-tune, `~5.4 MB`)
- **Task**: object detection (`task=detect`)
- **Framework**: Ultralytics (`ultralytics>=8.4.x`, đã test với `8.4.101`)
- **Input/Output màu**: BGR (khớp với OpenCV/Ultralytics, không cần convert)

Model không được tải xuống từ Internet. Nếu file thiếu, chương trình dừng với
thông báo rõ ràng hướng dẫn đặt model đúng vị trí.

## 2. Mapping 4 class

| class_id | class_name           | Ý nghĩa                                          |
|----------|----------------------|--------------------------------------------------|
| 0        | `subtitle`           | Phụ đề dịch/nội dung động ở dưới hoặc giữa màn hình |
| 1        | `watermark`          | Watermark (thường cố định, mờ)                    |
| 2        | `logo`               | Logo kênh/nhãn hiệu (thường cố định ở góc)         |
| 3        | `other_burn_in_text` | Text bị đốt vào video khác (UI, sticker, ...)      |

Nếu model load ra mapping khác, code **fail fast** (ném lỗi
`ClassMappingError`) thay vì chạy sai.

## 3. Vị trí model

```
media-flow/
├── models/
│   └── yolo/
│       └── best.pt
```

Thứ tự ưu tiên resolve đường dẫn:

1. Biến môi trường `MEDIA_FLOW_YOLO_WEIGHTS` (đường dẫn tuyệt đối hoặc tương đối với project root).
2. Giá trị `weights` trong `config/detection.yaml` (hoặc `providers.yaml` cho hybrid pipeline).
3. Mặc định `models/yolo/best.pt`.

Code dùng `pathlib.Path` nên hoạt động trên cả Windows và Linux.

## 4. Cách cấu hình

File chính: `config/detection.yaml`

```yaml
yolo:
  enabled: true
  weights: models/yolo/best.pt
  device: auto          # auto | cpu | cuda
  imgsz: 1280
  iou: 0.45
  max_det: 100
  half: auto            # auto = chỉ bật FP16 trên CUDA
  agnostic_nms: false
  verbose: false

  confidence:
    subtitle: 0.15
    watermark: 0.20
    logo: 0.20
    other_burn_in_text: 0.20

  temporal:
    enabled: true
    max_gap_frames: 2
    min_track_frames: 2
    iou_match_threshold: 0.40
    box_smoothing_alpha: 0.65
```

Có thể override qua biến môi trường: `MEDIA_FLOW_DETECTION__IMGSZ=640`,
`MEDIA_FLOW_DETECTION__DEVICE=cpu`, ...

Hybrid pipeline (OCR / region detector) đọc `weights` từ `config/providers.yaml`
(đã trỏ tới `models/yolo/best.pt`).

## 5. Chạy verify model

```bash
python scripts/verify_yolo_model.py [--weights PATH] [--device cpu|cuda|auto]
```

Script in ra: model path, file size, task, class mapping, device, versions của
Ultralytics/PyTorch, CUDA availability, rồi dry-run trên frame trắng. Exit code
khác 0 nếu model/config không hợp lệ.

PowerShell:

```powershell
python scripts\verify_yolo_model.py --device cpu
```

## 6. Test một video

```bash
python scripts/test_video_detection.py \
  --input path/to/input.mp4 \
  --output outputs/detected.mp4 \
  --conf-subtitle 0.15 \
  --show-labels \
  --save-json
```

PowerShell:

```powershell
python scripts\test_video_detection.py --input .\input\sample5.mp4 --output .\outputs\detected.mp4 --show-labels --save-json --device cuda
```

Các option:

- `--conf-<class> <float>`: override threshold theo class (`subtitle`, `watermark`, `logo`, `other_burn_in_text`).
- `--show-labels`: vẽ box + nhãn class/confidence.
- `--save-json`: ghi JSONL detection (1 dòng 1 detection).
- `--device cpu|cuda|auto`, `--imgsz N`, `--stride N`.

Video đầu ra **giữ nguyên FPS, resolution và audio** (mux lại bằng ffmpeg nếu
input có audio).

## 7. Cách chọn threshold

- **subtitle 0.15**: bắt được subtitle chạy nhanh / xuất hiện từng ký tự. Cần
  temporal filtering để loại false positive.
- **watermark / logo 0.20**: watermark và logo thường cố định, độ tin cậy cao.
- **other_burn_in_text 0.20**: UI text, sticker.
- Nếu thấy nhiều box nhiễu: tăng threshold của lớp đó lên `0.25–0.35`.
- Nếu miss subtitle: hạ `subtitle` xuống `0.10` (kèm temporal vẫn bật).

Không dùng một threshold cứng cho mọi lớp; mỗi lớp có ngưỡng riêng trong config.

## 8. Khác biệt giữa 4 lớp

| Lớp | Đặc điểm | Gợi ý xử lý |
|-----|----------|--------------|
| `subtitle` | Động, thay đổi theo thời gian, thường nằm dưới/giữa | Cần temporal stabilization; không dùng để che cố định |
| `watermark` | Cố định, mờ, nằm góc | Có thể tạo mask che cố định theo track dài |
| `logo` | Cố định, nét rõ | Mask che cố định, không merge với subtitle |
| `other_burn_in_text` | UI/sticker/khác | Xử lý tùy tình huống, không gộp chung |

## 9. Xử lý subtitle động

`TemporalStabilizer` (`src/infrastructure/detection/temporal.py`) thực hiện:

- match box giữa frame bằng IoU + class;
- giữ track qua gap 1–2 frame (`max_gap_frames=2`);
- làm mượt tọa độ bằng EMA (`box_smoothing_alpha=0.65`);
- không trộn track khác class;
- track biến mất sẽ bị drop nhanh (không tạo ghost box);
- tắt hoàn toàn bằng `temporal.enabled: false`.

## 10. CPU và CUDA

- `device: auto` → chọn CUDA nếu có, ngược lại CPU.
- CUDA: bật FP16 tự động (`half: auto`).
- CPU: luôn FP32 (không dùng FP16 trên CPU).
- Nếu cấu hình `cuda` mà không có CUDA → báo lỗi rõ ràng, không tự fallback im lặng.

## 11. Các lỗi thường gặp

| Lỗi | Nguyên nhân | Cách xử lý |
|-----|-------------|------------|
| `YOLO fine-tuned weights file was not found` | Model không nằm tại đường dẫn đã resolve | Đặt `best.pt` vào `models/yolo/` hoặc set `MEDIA_FLOW_YOLO_WEIGHTS` |
| `Model has N classes but 4 were expected` | Model khác (không phải 4 lớp burn-in) | Dùng đúng file fine-tune 4 lớp |
| `class names [...] do not match expected` | Mapping tên class khác | Kiểm tra lại file model |
| `CUDA is unavailable` | Cấu hình `device: cuda` trên máy không có GPU | Đổi `device: auto` hoặc `cpu` |
| `Blank/gray frame detected` | Video decode fail ở frame đó (codec) | Transcode video sang H.264 trước |

## 12. Thay model trong tương lai

1. Đặt file model mới vào `models/yolo/` (ví dụ `models/yolo/my_v2.pt`).
2. Cập nhật `config/detection.yaml` → `weights: models/yolo/my_v2.pt` (hoặc set `MEDIA_FLOW_YOLO_WEIGHTS`).
3. Nếu class mapping đổi: cập nhật `EXPECTED_CLASS_NAMES` trong `src/infrastructure/detection/schemas.py`.
4. Chạy `python scripts/verify_yolo_model.py`.
5. Không cần sửa pipeline; detector load theo config.

## 13. Format JSON output

Mỗi dòng JSONL là một detection:

```json
{
  "class_id": 0,
  "class_name": "subtitle",
  "confidence": 0.816406,
  "bbox_xyxy": [291.725, 929.525, 423.6875, 1060.8],
  "bbox_normalized": [0.405174, 0.726191, 0.588455, 0.82875],
  "frame_index": 3,
  "timestamp_seconds": 0.1,
  "frame_width": 720,
  "frame_height": 1280,
  "region_id": null,
  "group_id": null,
  "track_id": null
}
```

Bounding box luôn được clamp trong frame. Các trường `region_id/group_id/track_id`
được điền bởi post-processing/temporal khi có.

## 14. Sơ đồ pipeline

```
Frame (BGR)
   │
   ▼
YoloBurnInDetector.detect_frame / detect_batch
   │  (Ultralytics predict, torch.inference_mode, FP16 on CUDA)
   ▼
Detection (schema nội bộ, clamp bbox, không rò rỉ object Ultralytics)
   │
   ▼
Post-processing (per-class confidence → clamp → loại box nhỏ → dedupe → expand)
   │
   ▼
Temporal stabilization (IoU match, gap carry, EMA smoothing)
   │
   ▼
DetectionResult → JSON / drawing / ProposalBundle adapter (legacy pipeline)
```

## Module map

- `src/infrastructure/detection/paths.py` — resolve đường dẫn model (env > config > default).
- `src/infrastructure/detection/config.py` — `DetectionConfig`, `load_detection_config`.
- `src/infrastructure/detection/schemas.py` — `Detection`, `DetectionResult`, class mapping.
- `src/infrastructure/detection/base.py` — interface `BurnInDetector`.
- `src/infrastructure/detection/yolo_detector.py` — `YoloBurnInDetector`, `get_yolo_detector`.
- `src/infrastructure/detection/postprocess.py` — lọc/clamp/dedupe/expand/group.
- `src/infrastructure/detection/temporal.py` — `TemporalStabilizer`.
- `src/infrastructure/detection/adapter.py` — chuyển sang `ProposalBundle` cũ.
