from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_sweep_r1_remains_a_production_script():
    assert (ROOT / "scripts" / "sweep_r1.py").exists()


def test_sweep_r2_is_evaluation_only():
    assert not (ROOT / "scripts" / "sweep_r2.py").exists()
    assert (ROOT / "evaluation" / "sweep_r2.py").exists()


def test_legacy_sweeps_remain_archived():
    assert not (ROOT / "scripts" / "sweep_r3.py").exists()
    assert not (ROOT / "scripts" / "sweep_step7.py").exists()
