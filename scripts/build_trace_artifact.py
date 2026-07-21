import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def _load_gt(gt_path: Path) -> Dict[str, Any]:
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    if isinstance(gt, list):
        # Tests pass a bare list; wrap it
        gt = {"entries": gt}
    return gt


def build_artifact(source_path: Path, gt_path: Path, output_path: Path) -> None:
    # --- load & verify GT checksum ---
    sha256_path = gt_path.with_suffix(".sha256")
    if not sha256_path.exists():
        raise ValueError(f"frozen GT checksum mismatch: {sha256_path} not found")
    expected_hex = sha256_path.read_text(encoding="ascii").strip().split()[0]
    actual_hex = hashlib.sha256(gt_path.read_bytes()).hexdigest()
    if actual_hex != expected_hex:
        raise ValueError(f"frozen GT checksum mismatch: {actual_hex} != {expected_hex}")

    gt = _load_gt(gt_path)
    entries = {e["id"]: e["expected_is_subtitle"] for e in gt["entries"]}

    # --- source data ---
    with open(source_path, encoding="utf-8") as f:
        src = json.load(f)

    proposals = {p.get("proposal_id", p.get("id")): p for p in src.get("proposals", [])}
    tracks = {t["track_id"]: t for t in src.get("tracks", [])}
    classifications = [c for c in src.get("classifications", [])]
    class_by_id = {c["classification_id"]: c for c in classifications}
    regions = {r["region_id"]: r for r in src.get("regions", [])}
    ass_events = src.get("ass_events", [])

    # --- injective + surjective coverage ---
    covered = set()
    for c in classifications:
        gid = c["gt_id"]
        if gid in covered:
            raise ValueError("must cover frozen GT exactly once (duplicate gt_id)")
        covered.add(gid)
        if gid not in entries:
            raise ValueError("must cover frozen GT exactly once (unknown gt_id)")
    for gid in entries:
        if gid not in covered:
            raise ValueError("must cover frozen GT exactly once (missing coverage)")

    # --- build mask provenance chains ---
    chains = []
    for ev in ass_events:
        if ev.get("kind") != "mask":
            continue
        rid = ev.get("region_id")
        cid = ev.get("classification_id")
        if rid not in regions:
            raise ValueError(f"references unknown region_id: {rid}")
        reg = regions[rid]
        track_id = reg.get("track_id")
        if track_id is None or track_id not in tracks:
            raise ValueError(f"region {rid} references unknown track_id: {track_id}")
        trk = tracks[track_id]
        if cid not in class_by_id:
            raise ValueError(f"references unknown classification_id: {cid}")
        clf = class_by_id[cid]
        if not clf.get("is_subtitle"):
            raise ValueError(
                f"ASS mask event {ev['event_id']} does not have a positive classifier decision"
            )
        gid = clf["gt_id"]
        if not entries.get(gid):
            raise ValueError(
                f"ASS mask event {ev['event_id']} does not have a positive classifier decision"
            )
        chains.append({
            "ass_event_id": ev["event_id"],
            "region_id": rid,
            "classification_id": cid,
            "track_id": track_id,
            "proposal_ids": list(trk.get("proposal_ids", [])),
        })

    # --- confusion matrix ---
    tp = fp = tn = fn = 0
    for c in classifications:
        expected = entries[c["gt_id"]]
        actual = c["is_subtitle"]
        if expected and actual:
            tp += 1
        elif not expected and actual:
            fp += 1
        elif not expected and not actual:
            tn += 1
        else:  # expected but not actual
            fn += 1

    artifact = {
        "mask_provenance_chains": chains,
        "classifier_confusion": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        "ground_truth_sha256": actual_hex,
    }

    output_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")


if __name__ == "__main__":
    import sys
    src, gt, out = sys.argv[1:4]
    build_artifact(Path(src), Path(gt), Path(out))
