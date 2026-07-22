from __future__ import annotations
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class BenchmarkConfig:
    enabled: bool = True
    yolo_weights: str = "best.pt"
    input_videos: List[str] = field(default_factory=lambda: [
        'test_videos/real_mini_benchmark/sample1.mp4',
        'test_videos/real_mini_benchmark/sample2.mp4',
        'test_videos/real_mini_benchmark/sample3.mp4',
        'test_videos/real_mini_benchmark/sample4.mp4',
        'test_videos/real_mini_benchmark/sample5.mp4',
    ])
    output_dir: str = 'artifacts/hybrid_subtitle_benchmark'
    live_gemini: bool = False
    
    # YOLO Proposal thresholds
    yolo_conf: float = 0.05
    yolo_iou: float = 0.70
    yolo_imgsz: int = 1024
    
    # Grid/Ablation study parameters
    ablation_runs: List[Dict[str, Any]] = field(default_factory=lambda: [
        {'conf': 0.05, 'imgsz': 1024},
        {'conf': 0.10, 'imgsz': 1024},
        {'conf': 0.25, 'imgsz': 640},
        {'conf': 0.05, 'imgsz': 1280},
    ])
    
    # Processing FPS configs to measure trade-offs
    benchmark_fps_modes: List[float] = field(default_factory=lambda: [5.0, 10.0, -1.0]) # -1.0 means native FPS
    
    # Local grouping rules
    group_vertical_overlap_thresh: float = 0.3
    horizontal_gap_ratio: float = 2.0
    group_baseline_tolerance: float = 15.0
    
    # Temporal tracking
    track_iou_threshold: float = 0.25
    track_center_dist_thresh: float = 100.0
    track_max_gap_frames: int = 5
    
    # Gemini options
    gemini_model: str = DEFAULT_GEMINI_MODEL
    max_rep_frames: int = 5

    # Safe cover margins & strategy
    glyph_margin: float = 12.0
    outline_margin: float = 4.0
    shadow_margin: float = 2.0
    motion_margin: float = 5.0
    safe_cover_strategy: str = "percentile"
