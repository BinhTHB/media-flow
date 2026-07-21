import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "artifacts" / "round19" / "build_trace_artifact.py"


def load_module():
    spec = importlib.util.spec_from_file_location("round19_trace", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_builds_one_valid_chain_per_mask_and_confusion_matrix(tmp_path):
    module = load_module()
    gt = [
        {"id": "case-positive", "expected_is_subtitle": True},
        {"id": "case-negative", "expected_is_subtitle": False},
    ]
    gt_path = tmp_path / "frozen_gt.json"
    write_json(gt_path, gt)
    (tmp_path / "frozen_gt.sha256").write_text(
        hashlib.sha256(gt_path.read_bytes()).hexdigest() + "  frozen_gt.json\n", encoding="ascii"
    )
    source_path = tmp_path / "source.json"
    write_json(source_path, {
        "proposals": [{"proposal_id": "P1"}],
        "tracks": [{"track_id": "T1", "proposal_ids": ["P1"]}],
        "classifications": [
            {"classification_id": "C1", "track_id": "T1", "is_subtitle": True, "gt_id": "case-positive"},
            {"classification_id": "C2", "track_id": "T1", "is_subtitle": False, "gt_id": "case-negative"},
        ],
        "regions": [{"region_id": "R1", "track_id": "T1", "classification_id": "C1"}],
        "ass_events": [{"event_id": "A1", "kind": "mask", "region_id": "R1", "classification_id": "C1"}],
    })
    output_path = tmp_path / "trace.json"

    module.build_artifact(source_path, gt_path, output_path)

    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    assert artifact["mask_provenance_chains"] == [{
        "ass_event_id": "A1", "region_id": "R1", "classification_id": "C1",
        "track_id": "T1", "proposal_ids": ["P1"],
    }]
    assert artifact["classifier_confusion"] == {"TP": 1, "FP": 0, "TN": 1, "FN": 0}
    assert artifact["ground_truth_sha256"] == hashlib.sha256(gt_path.read_bytes()).hexdigest()


def test_rejects_mask_without_exactly_one_positive_provenance_chain(tmp_path):
    module = load_module()
    gt_path = tmp_path / "frozen_gt.json"
    write_json(gt_path, [{"id": "negative", "expected_is_subtitle": False}])
    (tmp_path / "frozen_gt.sha256").write_text(
        hashlib.sha256(gt_path.read_bytes()).hexdigest() + "  frozen_gt.json\n", encoding="ascii"
    )
    source_path = tmp_path / "source.json"
    write_json(source_path, {
        "proposals": [{"proposal_id": "P1"}],
        "tracks": [{"track_id": "T1", "proposal_ids": ["P1"]}],
        "classifications": [{"classification_id": "C1", "track_id": "T1", "is_subtitle": False, "gt_id": "negative"}],
        "regions": [{"region_id": "R1", "track_id": "T1", "classification_id": "C1"}],
        "ass_events": [{"event_id": "A1", "kind": "mask", "region_id": "R1", "classification_id": "C1"}],
    })

    with pytest.raises(ValueError, match="positive classifier"):
        module.build_artifact(source_path, gt_path, tmp_path / "trace.json")


def test_rejects_mutated_frozen_ground_truth(tmp_path):
    module = load_module()
    gt_path = tmp_path / "frozen_gt.json"
    write_json(gt_path, [{"id": "positive", "expected_is_subtitle": True}])
    (tmp_path / "frozen_gt.sha256").write_text("0" * 64 + "  frozen_gt.json\n", encoding="ascii")
    source_path = tmp_path / "source.json"
    write_json(source_path, {"proposals": [], "tracks": [], "classifications": [], "regions": [], "ass_events": []})

    with pytest.raises(ValueError, match="frozen GT checksum mismatch"):
        module.build_artifact(source_path, gt_path, tmp_path / "trace.json")


def test_rejects_classifier_results_that_do_not_cover_frozen_gt(tmp_path):
    module = load_module()
    gt_path = tmp_path / "frozen_gt.json"
    gt = [{"id": "positive", "expected_is_subtitle": True}, {"id": "negative", "expected_is_subtitle": False}]
    write_json(gt_path, gt)
    (tmp_path / "frozen_gt.sha256").write_text(
        hashlib.sha256(gt_path.read_bytes()).hexdigest() + "  frozen_gt.json\n", encoding="ascii"
    )
    source_path = tmp_path / "source.json"
    write_json(source_path, {"proposals": [], "tracks": [], "regions": [], "ass_events": [], "classifications": [
        {"classification_id": "C1", "track_id": "T1", "is_subtitle": True, "gt_id": "positive"}
    ]})

    with pytest.raises(ValueError, match="must cover frozen GT exactly once"):
        module.build_artifact(source_path, gt_path, tmp_path / "trace.json")


def test_rejects_duplicate_gt_id_in_classifications(tmp_path):
    """Same gt_id used twice -> uncovered duplicate leaves one gt entry unseen."""
    module = load_module()
    gt = [
        {"id": "pos", "expected_is_subtitle": True},
        {"id": "neg", "expected_is_subtitle": False},
    ]
    gt_path = tmp_path / "frozen_gt.json"
    write_json(gt_path, gt)
    (tmp_path / "frozen_gt.sha256").write_text(
        hashlib.sha256(gt_path.read_bytes()).hexdigest() + "  frozen_gt.json\n", encoding="ascii"
    )
    source_path = tmp_path / "source.json"
    write_json(source_path, {"proposals": [], "tracks": [], "regions": [], "ass_events": [], "classifications": [
        {"classification_id": "C1", "track_id": "T1", "is_subtitle": True, "gt_id": "pos"},
        {"classification_id": "C2", "track_id": "T1", "is_subtitle": True, "gt_id": "pos"},  # dup
    ]})

    with pytest.raises(ValueError, match="must cover frozen GT exactly once"):
        module.build_artifact(source_path, gt_path, tmp_path / "trace.json")


def test_rejects_duplicate_gt_id_even_when_all_gt_ids_are_present(tmp_path):
    module = load_module()
    gt = [
        {"id": "pos", "expected_is_subtitle": True},
        {"id": "neg", "expected_is_subtitle": False},
    ]
    gt_path = tmp_path / "frozen_gt.json"
    write_json(gt_path, gt)
    (tmp_path / "frozen_gt.sha256").write_text(
        hashlib.sha256(gt_path.read_bytes()).hexdigest() + "  frozen_gt.json\n", encoding="ascii"
    )
    source_path = tmp_path / "source.json"
    write_json(source_path, {"proposals": [], "tracks": [], "regions": [], "ass_events": [], "classifications": [
        {"classification_id": "C1", "track_id": "T1", "is_subtitle": True, "gt_id": "pos"},
        {"classification_id": "C2", "track_id": "T2", "is_subtitle": True, "gt_id": "pos"},
        {"classification_id": "C3", "track_id": "T3", "is_subtitle": False, "gt_id": "neg"},
    ]})

    with pytest.raises(ValueError, match="must cover frozen GT exactly once"):
        module.build_artifact(source_path, gt_path, tmp_path / "trace.json")


def test_rejects_mask_event_with_unknown_region_id(tmp_path):
    module = load_module()
    gt_path = tmp_path / "frozen_gt.json"
    write_json(gt_path, [{"id": "pos", "expected_is_subtitle": True}])
    (tmp_path / "frozen_gt.sha256").write_text(
        hashlib.sha256(gt_path.read_bytes()).hexdigest() + "  frozen_gt.json\n", encoding="ascii"
    )
    source_path = tmp_path / "source.json"
    write_json(source_path, {
        "proposals": [], "tracks": [{"track_id": "T1", "proposal_ids": []}],
        "classifications": [{"classification_id": "C1", "track_id": "T1", "is_subtitle": True, "gt_id": "pos"}],
        "regions": [],
        "ass_events": [{"event_id": "A1", "kind": "mask", "region_id": "does_not_exist", "classification_id": "C1"}],
    })

    with pytest.raises(ValueError, match="references unknown region_id"):
        module.build_artifact(source_path, gt_path, tmp_path / "trace.json")


def test_rejects_ass_event_classification_with_is_subtitle_true_but_gt_expected_false(tmp_path):
    """Classifier says subtitle but GT says non-subtitle -> fail mask."""
    module = load_module()
    gt_path = tmp_path / "frozen_gt.json"
    write_json(gt_path, [{"id": "neg", "expected_is_subtitle": False}])
    (tmp_path / "frozen_gt.sha256").write_text(
        hashlib.sha256(gt_path.read_bytes()).hexdigest() + "  frozen_gt.json\n", encoding="ascii"
    )
    source_path = tmp_path / "source.json"
    write_json(source_path, {
        "proposals": [{"proposal_id": "P1"}],
        "tracks": [{"track_id": "T1", "proposal_ids": ["P1"]}],
        "classifications": [{"classification_id": "C1", "track_id": "T1", "is_subtitle": True, "gt_id": "neg"}],
        "regions": [{"region_id": "R1", "track_id": "T1", "classification_id": "C1"}],
        "ass_events": [{"event_id": "A1", "kind": "mask", "region_id": "R1", "classification_id": "C1"}],
    })

    with pytest.raises(ValueError, match="does not have a positive classifier decision"):
        module.build_artifact(source_path, gt_path, tmp_path / "trace.json")


def test_empty_source_produces_correct_confusion_zeros(tmp_path):
    """No classifications: TP=FP=TN=FN=0, no mask chains."""
    module = load_module()
    gt_path = tmp_path / "frozen_gt.json"
    gt = [{"id": "pos", "expected_is_subtitle": True}]
    write_json(gt_path, gt)
    (tmp_path / "frozen_gt.sha256").write_text(
        hashlib.sha256(gt_path.read_bytes()).hexdigest() + "  frozen_gt.json\n", encoding="ascii"
    )
    source_path = tmp_path / "source.json"
    write_json(source_path, {"proposals": [], "tracks": [], "classifications": [], "regions": [], "ass_events": []})

    with pytest.raises(ValueError, match="must cover frozen GT exactly once"):
        module.build_artifact(source_path, gt_path, tmp_path / "trace.json")
