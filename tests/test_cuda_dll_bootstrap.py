import os
from pathlib import Path

from src.infrastructure.runtime import cuda_dll_bootstrap


def test_registers_existing_nvidia_dll_directories_on_windows(monkeypatch, tmp_path):
    site_packages = tmp_path / "site-packages"
    expected = [
        site_packages / "nvidia" / "cublas" / "bin",
        site_packages / "nvidia" / "cudnn" / "bin",
        site_packages / "nvidia" / "cuda_runtime" / "bin",
        site_packages / "nvidia" / "cuda_nvrtc" / "bin",
    ]
    for path in expected:
        path.mkdir(parents=True)

    handles = []
    monkeypatch.setattr(cuda_dll_bootstrap.os, "name", "nt")
    monkeypatch.setattr(cuda_dll_bootstrap.site, "getsitepackages", lambda: [str(site_packages)])
    monkeypatch.setattr(cuda_dll_bootstrap.os, "add_dll_directory", lambda path: handles.append(path) or object())
    cuda_dll_bootstrap._DLL_HANDLES.clear()

    cuda_dll_bootstrap.register_cuda_dll_directories()

    assert handles == [os.fspath(path) for path in expected]
    assert len(cuda_dll_bootstrap._DLL_HANDLES) == 4


def test_is_noop_outside_windows(monkeypatch):
    monkeypatch.setattr(cuda_dll_bootstrap.os, "name", "posix")
    monkeypatch.setattr(
        cuda_dll_bootstrap.os,
        "add_dll_directory",
        lambda _path: (_ for _ in ()).throw(AssertionError("must not register")),
        raising=False,
    )

    cuda_dll_bootstrap.register_cuda_dll_directories()


def test_skips_missing_packages_on_windows(monkeypatch, tmp_path):
    monkeypatch.setattr(cuda_dll_bootstrap.os, "name", "nt")
    monkeypatch.setattr(cuda_dll_bootstrap.site, "getsitepackages", lambda: [str(tmp_path)])
    monkeypatch.setattr(cuda_dll_bootstrap.os, "add_dll_directory", lambda _path: object())
    cuda_dll_bootstrap._DLL_HANDLES.clear()

    cuda_dll_bootstrap.register_cuda_dll_directories()

    assert cuda_dll_bootstrap._DLL_HANDLES == []


def test_provider_registers_cuda_dlls_before_import(monkeypatch):
    from src.infrastructure.providers.speech import faster_whisper_speech_provider as provider_module

    events = []
    monkeypatch.setattr(provider_module, "register_cuda_dll_directories", lambda: events.append("register"))

    class WhisperModel:
        def __init__(self, *_args, **_kwargs):
            events.append("model")

    import sys
    import types
    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=WhisperModel))

    provider_module.FasterWhisperSpeechProvider(device="cuda")._get_model()

    assert events == ["register", "model"]
