"""Filesystem/AST architecture guards, including untracked production files."""

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
FORBIDDEN_INFRASTRUCTURE_PACKAGES = {
    "cv2",
    "faster_whisper",
    "ffmpeg",
    "google",
    "moviepy",
    "openai",
    "paddleocr",
    "rapidocr",
    "ultralytics",
    "whisper",
}
PRODUCTION_ONLY_FORBIDDEN = ("benchmark", "benchmarks", "evaluation", "evaluations", "scripts")


def _python_files(root: Path):
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def _imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def _matches(module: str, prefixes) -> bool:
    return any(module == prefix or module.startswith(prefix + ".") for prefix in prefixes)


def _assert_boundary(root: Path, forbidden_local=(), forbid_external=False):
    violations = {}
    for path in _python_files(root):
        bad = []
        for module in _imports(path):
            if _matches(module, forbidden_local):
                bad.append(module)
            if forbid_external and module.split(".")[0] in FORBIDDEN_INFRASTRUCTURE_PACKAGES:
                bad.append(module)
        if bad:
            violations[str(path.relative_to(PROJECT_ROOT))] = sorted(set(bad))
    assert not violations, violations


def test_core_boundary():
    _assert_boundary(
        SRC_ROOT / "core",
        ("src.application", "src.features", "src.infrastructure", "src.interfaces"),
        forbid_external=True,
    )


def test_application_boundary():
    _assert_boundary(
        SRC_ROOT / "application",
        ("src.infrastructure", "src.interfaces"),
        forbid_external=True,
    )


def test_features_depend_on_ports_not_infrastructure_or_interfaces():
    _assert_boundary(
        SRC_ROOT / "features",
        ("src.infrastructure", "src.interfaces"),
    )


def test_audio_and_visual_features_are_independent():
    _assert_boundary(SRC_ROOT / "features" / "audio", ("src.features.visual",))
    _assert_boundary(SRC_ROOT / "features" / "visual", ("src.features.audio",))


def test_production_does_not_import_benchmarks_evaluations_or_scripts():
    _assert_boundary(SRC_ROOT, PRODUCTION_ONLY_FORBIDDEN)
