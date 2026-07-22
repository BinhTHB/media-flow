# Hybrid Subtitle Detector - Ground Truth Benchmark

Chay tren cac frame co san trong `test_videos/subtitle_replacement_benchmark/`.
Doi chieu region du doan voi ground truth theo IoU >= 0.5 tai tung checkpoint.

| set | checkpoints | hits | miss | hit_rate | mean_iou |
| --- | --- | --- | --- | --- | --- |
| bottom_stable | 6 | 6 | 0 | 1.0 | 0.9248 |
| moving | 6 | 0 | 6 | 0.0 | 0.0 |

## Ghi chu
- Set `moving` dung median box ca track nen IoU thap la gioi han da biet cua heuristic offline.
- Danh gia nghiem tuc hon can Gemini layer (live verification).
