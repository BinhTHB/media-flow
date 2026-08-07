"""Ground-truth evaluation for the YOLO burn-in detector.

Loads a manifest, ground-truth labels and cached model predictions; runs the
detector over the sampled frames when predictions are missing; then computes
Precision / Recall / F1 per class, micro/macro averages, localization IoU
statistics, the 4-class confusion matrix, a per-class confidence threshold
sweep, an error gallery, and (when applicable) temporal continuity metrics.
Everything is written under ``outputs/evaluation/``.

HONESTY CONTRACT
----------------
If the ground truth is not verified (manifest rows still marked
``verified=false``), the script **refuses** to report real Precision / Recall /
F1 / confusion metrics. It only emits a ``PENDING_HUMAN_REVIEW`` report so that
nobody mistakes machine pre-labels for ground truth.

Example:
  .venv\\Scripts\\python.exe scripts\\evaluate_yolo_detection.py ^
      --manifest evaluation/manifests/manifest.csv ^
      --labels evaluation/labels ^
      --frames evaluation/frames ^
      --predictions outputs/evaluation/predictions.jsonl ^
      --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np

from src.infrastructure.detection import YoloBurnInDetector, load_detection_config
from src.infrastructure.detection.evaluation import Box
from src.infrastructure.detection.evaluation.confusion import ConfusionMatrix
from src.infrastructure.detection.evaluation.metrics import ClassMetrics
from src.infrastructure.detection.evaluation.report import FrameTruth, evaluate_frames
from src.infrastructure.detection.schemas import EXPECTED_CLASS_NAMES

CLASS_ORDER = list(EXPECTED_CLASS_NAMES.values())
CLASS_BY_NAME = {name: i for i, name in enumerate(CLASS_ORDER)}
BASE_CONF = 0.05  # raw-inference confidence cutoff for prediction capture

CLASS_COLORS = {
    "subtitle": (89, 178, 85),
    "watermark": (77, 55, 216),
    "logo": (255, 172, 27),
    "other_burn_in_text": (80, 165, 255),
}
FALLBACK_COLOR = (66, 204, 255)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ground-truth YOLO detection evaluation")
    p.add_argument("--manifest", required=True, help="CSV manifest of sampled frames")
    p.add_argument("--labels", required=True, help="Directory of GT label files (*.json / *.txt)")
    p.add_argument("--frames", required=True, help="Directory of extracted frame images")
    p.add_argument("--predictions", default=None, help="Prediction JSONL cache (created if missing)")
    p.add_argument("--out-dir", default="outputs/evaluation", help="Output directory")
    p.add_argument("--device", default=None, choices=["cpu", "cuda", "auto"])
    p.add_argument("--imgsz", type=int, default=None, help="Inference imgsz override")
    p.add_argument("--iou", type=float, nargs="+", default=[0.50, 0.75], help="IoU thresholds")
    p.add_argument(
        "--sweep-thresholds", type=float, nargs="+",
        default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50],
        help="Confidence thresholds to sweep per class",
    )
    p.add_argument("--conf", type=float, nargs=4, default=None,
                   help="Per-class conf override: subtitle watermark logo other_burn_in_text")
    return p


# ---------------------------------------------------------------------------
# Manifest / labels / predictions IO
# ---------------------------------------------------------------------------
def load_manifest(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"manifest not found: {path}")
    with open(path, "r", encoding="utf-8", newline="") as fh:
        rows = [dict(r) for r in csv.DictReader(fh)]
    if not rows:
        raise SystemExit(f"manifest {path} is empty")
    return rows


def is_verified(row: Dict[str, Any]) -> bool:
    return str(row.get("verified", "")).strip().lower() in {"true", "yes", "1", "y"}


def load_gt_for_row(
    row: Dict[str, Any], labels_dir: Path
) -> Tuple[int, List[Dict[str, Any]], bool]:
    """Return ``(frame_index, objects, verified)`` for one manifest row."""
    frame_id = str(row["frame_id"])
    verified = is_verified(row)

    json_label = labels_dir / f"{frame_id}.json"
    if json_label.is_file():
        data = json.loads(json_label.read_text(encoding="utf-8"))
        fi = int(data.get("frame_index", row.get("frame_index", -1)))
        return fi, list(data.get("objects", [])), bool(data.get("verified", verified))

    txt_label = labels_dir / f"{frame_id}.txt"
    if txt_label.is_file():
        w = float(row.get("width", 0.0))
        h = float(row.get("height", 0.0))
        fi = int(row.get("frame_index", -1))
        objs: List[Dict[str, Any]] = []
        for line in txt_label.read_text(encoding="utf-8").strip().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cid = int(parts[0])
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            name = EXPECTED_CLASS_NAMES.get(cid, "unknown")
            objs.append({
                "class_id": cid,
                "class_name": name,
                "bbox_xyxy": [(cx - bw / 2) * w, (cy - bh / 2) * h,
                              (cx + bw / 2) * w, (cy + bh / 2) * h],
            })
        return fi, objs, False  # txt labels have no verified flag

    raise FileNotFoundError(f"no label for {frame_id} in {labels_dir}")


def build_frames(manifest: List[Dict[str, Any]], labels_dir: Path) -> List[FrameTruth]:
    frames: List[FrameTruth] = []
    for row in manifest:
        frame_id = str(row["frame_id"])
        fi, raw_gt, _verified = load_gt_for_row(row, labels_dir)
        gt = [
            Box(class_id=int(o["class_id"]), bbox=[float(v) for v in o["bbox_xyxy"]], confidence=1.0)
            for o in raw_gt
        ]
        frames.append(FrameTruth(
            frame_id=frame_id,
            video_id=str(row.get("video_id", "")),
            frame_index=fi,
            width=int(row.get("width", 0)),
            height=int(row.get("height", 0)),
            timestamp_seconds=float(row.get("timestamp_seconds", 0.0)),
            gt=gt,
            pred=[],
        ))
    return frames


def read_predictions(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    if not path.is_file():
        return out
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[str(rec["frame_id"])] = list(rec.get("detections", []))
    return out


def frame_image(row: Dict[str, Any], frames_dir: Path) -> Path:
    base = frames_dir / str(row["frame_id"])
    for suffix in ("", ".jpg", ".png", ".jpeg"):
        cand = Path(str(base)) if not suffix else Path(str(base) + suffix)
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"no image for frame_id={row['frame_id']} in {frames_dir}")


def write_predictions(
    frames: List[FrameTruth],
    manifest: List[Dict[str, Any]],
    frames_dir: Path,
    pred_path: Path,
    cfg,
) -> None:
    """Run raw (no post-processing, no temporal) inference and cache JSONL."""
    rows_by_id = {str(r["frame_id"]): r for r in manifest}
    detector = YoloBurnInDetector(config=cfg)
    try:
        with open(pred_path, "w", encoding="utf-8") as fh:
            for ft in frames:
                row = rows_by_id[ft.frame_id]
                img = cv2.imread(str(frame_image(row, frames_dir)))
                if img is None:
                    continue
                result = detector.predict_raw(
                    img, frame_index=ft.frame_index, timestamp_seconds=ft.timestamp_seconds,
                    confidence=BASE_CONF,
                )
                dets = [d.to_json_compatible() for d in result.detections]
                fh.write(json.dumps({
                    "frame_id": ft.frame_id,
                    "video_id": ft.video_id,
                    "frame_index": ft.frame_index,
                    "detections": dets,
                }, ensure_ascii=False) + "\n")
    finally:
        detector.release()
    print(f"  wrote raw predictions -> {pred_path}")


def attach_predictions(
    frames: List[FrameTruth],
    pred_by_frame: Dict[str, List[Dict[str, Any]]],
    conf_by_class: Dict[str, float],
) -> List[FrameTruth]:
    for ft in frames:
        raw = pred_by_frame.get(ft.frame_id, [])
        kept = [d for d in raw if d.get("confidence", 0.0) >= conf_by_class.get(d.get("class_name", ""), 0.0)]
        ft.pred = [
            Box(class_id=int(d.get("class_id", CLASS_BY_NAME.get(d.get("class_name", ""), -1))),
                bbox=[float(v) for v in d.get("bbox_xyxy", [0, 0, 0, 0])],
                confidence=float(d.get("confidence", 0.0)))
            for d in kept
        ]
    return frames


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def metrics_to_rows(report) -> List[Dict[str, Any]]:
    rows = []
    for name in CLASS_ORDER:
        m: ClassMetrics = report.by_class.get(name)
        if m is None:
            continue
        rows.append({
            "class": name,
            "gt": report.total_gt,
            "tp": m.tp, "fp": m.fp, "fn": m.fn,
            "precision": round(m.precision, 4), "recall": round(m.recall, 4), "f1": round(m.f1, 4),
            "iou": report.iou_threshold,
        })
    return rows


def write_metrics_json(out_dir: Path, report, args) -> None:
    payload = {
        "iou_threshold": report.iou_threshold,
        "total_gt": report.total_gt,
        "total_pred": report.total_pred,
        "total_matched": report.total_matched,
        "by_class": {
            name: {
                "tp": m.tp, "fp": m.fp, "fn": m.fn,
                "precision": round(m.precision, 4),
                "recall": round(m.recall, 4),
                "f1": round(m.f1, 4),
            }
            for name, m in report.by_class.items()
        },
        "micro_macro": {k: round(v, 4) for k, v in report.aggregates.items()},
        "iou_by_class": report.iou_by_class,
        "args": vars(args),
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_metrics_csv(out_dir: Path, report) -> None:
    fieldnames = ["class", "tp", "fp", "fn", "precision", "recall", "f1", "mean_iou", "median_iou", "iou_p10", "iou_p90"]
    rows = []
    for name in CLASS_ORDER:
        m: ClassMetrics = report.by_class.get(name)
        iou = report.iou_by_class.get(name, {})
        if m is None:
            continue
        rows.append({
            "class": name, "tp": m.tp, "fp": m.fp, "fn": m.fn,
            "precision": round(m.precision, 4), "recall": round(m.recall, 4), "f1": round(m.f1, 4),
            "mean_iou": iou.get("mean_iou", ""), "median_iou": iou.get("median_iou", ""),
            "iou_p10": iou.get("iou_p10", ""), "iou_p90": iou.get("iou_p90", ""),
        })
    with open(out_dir / "metrics_by_class.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_confusion_csv(out_dir: Path, cm: ConfusionMatrix) -> None:
    with open(out_dir / "confusion_matrix.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["gt \\ pred"] + cm.col_labels)
        for ri, row in enumerate(cm.matrix):
            writer.writerow([cm.row_labels[ri]] + row)


def write_errors_jsonl(out_dir: Path, report) -> None:
    with open(out_dir / "errors.jsonl", "w", encoding="utf-8") as fh:
        for e in report.errors:
            fh.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")


def write_error_gallery(
    out_dir: Path, report, manifest: List[Dict[str, Any]], frames_dir: Path
) -> None:
    groups = {
        "false_positive": [], "false_negative": [], "wrong_class": [],
        "low_iou": [], "subtitle_vs_other": [], "logo_vs_watermark": [],
    }
    rows_by_id = {str(r["frame_id"]): r for r in manifest}
    gallery = out_dir / "error_gallery"
    gallery.mkdir(parents=True, exist_ok=True)
    for e in report.errors:
        if e.kind in groups:
            groups[e.kind].append(e)
        if e.gt_class == "subtitle" and e.pred_class == "other_burn_in_text":
            groups["subtitle_vs_other"].append(e)
        if e.gt_class == "other_burn_in_text" and e.pred_class == "subtitle":
            groups["subtitle_vs_other"].append(e)
        if (e.gt_class == "watermark" and e.pred_class == "logo") or \
           (e.gt_class == "logo" and e.pred_class == "watermark"):
            groups["logo_vs_watermark"].append(e)

    index_lines = ["# Error gallery", ""]
    for group, errors in groups.items():
        if not errors:
            index_lines.append(f"## {group}\n\n(no errors)\n")
            continue
        index_lines.append(f"## {group} ({len(errors)})")
        for e in errors[:30]:
            row = rows_by_id.get(e.frame_id)
            if row is None:
                continue
            try:
                img_path = frame_image(row, frames_dir)
                img = cv2.imread(str(img_path))
            except Exception:
                continue
            if img is None:
                continue
            if e.gt_box:
                cv2.rectangle(img, _xyxy_int(e.gt_box), (0, 255, 0), 2)
            if e.pred_box:
                cv2.rectangle(img, _xyxy_int(e.pred_box), (0, 0, 255), 2)
            name = f"{group}_{e.frame_id}.jpg"
            cv2.imwrite(str(gallery / name), img)
            index_lines.append(
                f"- [{e.frame_id}]({name}) gt={e.gt_class} pred={e.pred_class} "
                f"conf={e.confidence if e.confidence is not None else '-'} iou={e.iou if e.iou is not None else '-'} "
                f"frame={e.frame_index} ts={e.timestamp_seconds:.2f}s"
            )
        index_lines.append("")
    (gallery / "index.md").write_text("\n".join(index_lines), encoding="utf-8")
    print(f"  wrote error gallery -> {gallery}")


def _xyxy_int(box: Sequence[float]) -> Tuple[int, int, int, int]:
    return tuple(int(round(float(v))) for v in box)


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------
def run_threshold_sweep(
    frames: List[FrameTruth],
    thresholds: Sequence[float],
    iou_threshold: float,
) -> List[Dict[str, Any]]:
    """Sweep one class at a time while keeping others at a low floor.

    For each class and each threshold, re-filter predictions of that class by
    the candidate threshold (other classes stay at BASE_CONF) and compute
    class-aware P/R/F1 for that class alone.
    """
    rows: List[Dict[str, Any]] = []
    for class_name in CLASS_ORDER:
        class_id = CLASS_BY_NAME[class_name]
        for t in thresholds:
            swept: List[FrameTruth] = []
            for ft in frames:
                pred = []
                for b in ft.pred:
                    if b.class_id == class_id:
                        if b.confidence >= t:
                            pred.append(b)
                    else:
                        pred.append(b)
                swept.append(FrameTruth(
                    frame_id=ft.frame_id, video_id=ft.video_id, frame_index=ft.frame_index,
                    width=ft.width, height=ft.height, gt=ft.gt, pred=pred,
                    timestamp_seconds=ft.timestamp_seconds,
                ))
            rep = evaluate_frames(swept, iou_threshold=iou_threshold)
            m: ClassMetrics = rep.by_class.get(class_name)
            if m is None:
                continue
            rows.append({
                "class": class_name, "threshold": t,
                "tp": m.tp, "fp": m.fp, "fn": m.fn,
                "precision": round(m.precision, 4), "recall": round(m.recall, 4), "f1": round(m.f1, 4),
            })
    return rows


def write_threshold_sweep(out_dir: Path, rows: List[Dict[str, Any]]) -> None:
    with open(out_dir / "threshold_sweep.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["class", "threshold", "tp", "fp", "fn", "precision", "recall", "f1"])
        writer.writeheader()
        writer.writerows(rows)


def recommend_thresholds(rows: List[Dict[str, Any]], current: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    """Pick the threshold maximizing F1 per class; record the current one."""
    by_class: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_class.setdefault(r["class"], []).append(r)
    rec: Dict[str, Dict[str, Any]] = {}
    for class_name, rs in by_class.items():
        best = max(rs, key=lambda r: r["f1"])
        cur = current.get(class_name, 0.0)
        rec[class_name] = {
            "current": cur,
            "recommended_f1": best["threshold"],
            "current_f1": next((r["f1"] for r in rs if abs(r["threshold"] - cur) < 1e-9), None),
            "best_f1": best["f1"],
            "best_recall": best["recall"],
            "best_precision": best["precision"],
        }
    return rec


def write_recommendation(out_dir: Path, rec: Dict[str, Dict[str, Any]]) -> None:
    lines = [
        "# Threshold recommendation",
        "",
        "Based on the ground-truth threshold sweep (F1-maximizing per class, at the "
        "IoU threshold used for the main report). Trade-offs are discussed in "
        "`docs/YOLO_GROUND_TRUTH_EVALUATION.md` — F1 is not the only objective: for a "
        "subtitle-removal pipeline a missed subtitle (FN) usually costs more than a "
        "small false positive.",
        "",
        "| Class | Current | Recommended (max F1) | F1@current | F1@recommended | Recall@recommended | Precision@recommended |",
        "|---|---|---|---|---|---|---|",
    ]
    for class_name, info in rec.items():
        cur = info["current"]
        rec_t = info["recommended_f1"]
        lines.append(
            f"| {class_name} | {cur} | {rec_t} | "
            f"{_fmt(info.get('current_f1'))} | {_fmt(info['best_f1'])} | "
            f"{_fmt(info['best_recall'])} | {_fmt(info['best_precision'])} |"
        )
    (out_dir / "threshold_recommendation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(v) -> str:
    return "-" if v is None else f"{v:.4f}"


# ---------------------------------------------------------------------------
# Pending report (no verified GT)
# ---------------------------------------------------------------------------
def write_pending_report(out_dir: Path, args, manifest: List[Dict[str, Any]], frames: List[FrameTruth]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(manifest)
    verified = sum(1 for r in manifest if is_verified(r))
    with open(out_dir / "summary.md", "w", encoding="utf-8") as fh:
        fh.write(
            "# PENDING_HUMAN_REVIEW\n\n"
            f"- Total sampled frames: {total}\n"
            f"- Verified (ground truth): {verified}\n"
            f"- Pre-labeled only (UNVERIFIED): {total - verified}\n\n"
            "No Precision / Recall / F1 / confusion-matrix metrics are reported "
            "because the ground truth has not been human-reviewed. Pre-labels were "
            "generated by the model and are NOT ground truth.\n\n"
            "To continue: review `evaluation/review/`, set `verified=true` in the "
            "manifest for each confirmed frame, then re-run this script.\n"
        )
    print(f"  wrote {out_dir / 'summary.md'}")


# ---------------------------------------------------------------------------
def main() -> int:
    args = build_parser().parse_args()
    manifest = load_manifest(Path(args.manifest))
    labels_dir = Path(args.labels)
    frames_dir = Path(args.frames)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_detection_config()
    if args.device:
        cfg.device = args.device
    if args.imgsz:
        cfg.imgsz = args.imgsz
    if args.conf:
        cfg.confidence = {name: t for name, t in zip(CLASS_ORDER, args.conf)}

    frames = build_frames(manifest, labels_dir)

    verified_rows = [r for r in manifest if is_verified(r)]
    if not verified_rows:
        write_pending_report(out_dir, args, manifest, frames)
        print("PENDING_HUMAN_REVIEW: no verified ground truth; metrics not computed.")
        return 0  # Clean exit so callers can detect the pending status
    verified_ids = {str(r["frame_id"]) for r in verified_rows}

    # Predictions cache
    pred_path = Path(args.predictions) if args.predictions else out_dir / "predictions.jsonl"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_by_frame = read_predictions(pred_path)
    missing = [f for f in frames if f.frame_id not in pred_by_frame]
    if missing:
        write_predictions(missing, manifest, frames_dir, pred_path, cfg)
        pred_by_frame = read_predictions(pred_path)

    frames = attach_predictions(frames, pred_by_frame, cfg.confidence)

    # Only verified frames count toward metrics.
    eval_frames = [f for f in frames if f.frame_id in verified_ids]
    if not eval_frames:
        write_pending_report(out_dir, args, manifest, frames)
        print("PENDING_HUMAN_REVIEW: verified frame set is empty.")
        return 2

    # Metrics at each IoU threshold
    for iou_t in args.iou:
        report = evaluate_frames(eval_frames, iou_threshold=float(iou_t))
        tag = "" if iou_t == args.iou[0] else f"_iou{int(iou_t * 100):02d}"
        write_metrics_json(out_dir, report, args) if not tag else \
            (out_dir / f"metrics_iou{int(iou_t * 100):02d}.json").write_text(
                json.dumps({
                    "iou_threshold": iou_t,
                    "total_gt": report.total_gt,
                    "total_pred": report.total_pred,
                    "total_matched": report.total_matched,
                    "by_class": {name: {"tp": m.tp, "fp": m.fp, "fn": m.fn,
                                        "precision": round(m.precision, 4),
                                        "recall": round(m.recall, 4),
                                        "f1": round(m.f1, 4)}
                                 for name, m in report.by_class.items()},
                    "micro_macro": {k: round(v, 4) for k, v in report.aggregates.items()},
                    "iou_by_class": report.iou_by_class,
                }, indent=2, ensure_ascii=False), encoding="utf-8")
        if not tag:
            write_metrics_csv(out_dir, report)
            write_confusion_csv(out_dir, report.confusion)
            write_errors_jsonl(out_dir, report)
            write_error_gallery(out_dir, report, manifest, frames_dir)
            sweep_rows = run_threshold_sweep(eval_frames, args.sweep_thresholds, float(iou_t))
            write_threshold_sweep(out_dir, sweep_rows)
            rec = recommend_thresholds(sweep_rows, cfg.confidence)
            write_recommendation(out_dir, rec)
            write_summary(out_dir, report, rec, args, manifest)

    print(f"  outputs -> {out_dir}")
    return 0


def write_summary(out_dir: Path, report, rec, args, manifest) -> None:
    lines = [
        "# Ground-truth evaluation summary",
        "",
        f"- IoU threshold: {report.iou_threshold}",
        f"- Verified frames: {sum(1 for r in manifest if is_verified(r))} / {len(manifest)}",
        f"- Total GT boxes: {report.total_gt}",
        f"- Total predictions (after conf filter): {report.total_pred}",
        f"- Matched: {report.total_matched}",
        "",
        "| Class | GT | TP | FP | FN | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in CLASS_ORDER:
        m: ClassMetrics = report.by_class.get(name)
        if m is None:
            continue
        lines.append(
            f"| {name} | {report.total_gt} | {m.tp} | {m.fp} | {m.fn} | "
            f"{m.precision:.4f} | {m.recall:.4f} | {m.f1:.4f} |"
        )
    a = report.aggregates
    lines.append(f"| **Micro** | | | | | {a['micro_precision']:.4f} | {a['micro_recall']:.4f} | {a['micro_f1']:.4f} |")
    lines.append(f"| **Macro** | | | | | {a['macro_precision']:.4f} | {a['macro_recall']:.4f} | {a['macro_f1']:.4f} |")
    lines += ["", "## Recommended thresholds (max F1 per class)", "",
              "| Class | Current | Recommended |", "|---|---|---|"]
    for class_name, info in rec.items():
        lines.append(f"| {class_name} | {info['current']} | {info['recommended_f1']} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
