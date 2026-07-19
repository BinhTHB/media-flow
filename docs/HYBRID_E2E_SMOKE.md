# Hybrid Subtitle Detector - E2E Smoke Test

## CLI Command run
```bash
python src/interfaces/cli/demo.py test_videos/subtitle_replacement_benchmark/bottom_stable/video.mp4 --subtitle --subtitle-placement --placement-mode replace_source_subtitle --video-provider ffmpeg --frame-provider ffmpeg --frame-interval 0.5 --subtitle-region-detector hybrid_yolo --output-dir output/hybrid_e2e --job-id e2e_hybrid
```

Number of regions detected: 1

## Checkpoint IoU evaluation
| timestamp | best_iou | hit |
| --- | --- | --- |
| 0.0 | 0.0 | False |
| 0.5 | 0.9239 | True |
| 1.0 | 0.9239 | True |
| 1.5 | 0.9239 | True |
| 2.0 | 0.9239 | True |
| 2.5 | 0.9239 | True |

**Hits (IoU >= 0.5):** 5 / 6

**Hit Rate:** 0.8333

**Mean IoU:** 0.7699

## Conclusion
The hybrid_yolo region detector passes the end-to-end integration smoke test.
It successfully identifies the source subtitles at the bottom of the video frames.
