import sys
import types

import pytest

from scripts.targeted_ocr_localization import _argument_parser, _validate_ocr_device
from src.infrastructure.runtime.ocr_device_policy import validate_ocr_device
from src.shared.exceptions import ProviderError


def _paddle(cuda: bool, device: str):
    return types.SimpleNamespace(
        device=types.SimpleNamespace(
            is_compiled_with_cuda=lambda: cuda,
            get_device=lambda: device,
        )
    )


def test_targeted_ocr_default_device_is_gpu():
    assert _argument_parser().parse_args(["artifact", "output"]).ocr_device == "gpu"


def test_gpu_request_fails_without_cuda(monkeypatch):
    monkeypatch.setitem(sys.modules, "paddle", _paddle(False, "cpu"))

    with pytest.raises(ProviderError, match="OCR_DEVICE_GPU_REQUIRED_BUT_UNAVAILABLE"):
        validate_ocr_device("gpu")


def test_gpu_request_does_not_silently_fallback_to_cpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "paddle", _paddle(True, "cpu"))

    with pytest.raises(RuntimeError, match="OCR_DEVICE_GPU_REQUIRED_BUT_UNAVAILABLE"):
        _validate_ocr_device("gpu")


def test_explicit_cpu_is_allowed_and_reported():
    report = _validate_ocr_device("cpu")

    assert report["ocr_device_requested"] == "cpu"
    assert report["ocr_device_used"] == "cpu"
    assert report["fallback_to_cpu"] is False
    assert report["fallback_reason"] is None
    assert report["python_executable"] == sys.executable
