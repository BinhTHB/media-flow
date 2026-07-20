import os
import site
import ctypes
import logging

logger = logging.getLogger(__name__)

_DLL_HANDLES = []
_LOADED_DLL_HANDLES = {}
_CUDA_DLL_DIRECTORIES = (
    ("nvidia", "cublas", "bin"),
    ("nvidia", "cudnn", "bin"),
    ("nvidia", "cuda_runtime", "bin"),
    ("nvidia", "cuda_nvrtc", "bin"),
)

_PRELOAD_DLLS = (
    "cudart64_12.dll",
    "cublasLt64_12.dll",
    "cublas64_12.dll",
    "cudnn64_9.dll",
    "nvrtc64_120_0.dll",
)


def register_cuda_dll_directories() -> None:
    if os.name != "nt":
        return

    # 1. Register DLL directories
    for site_packages in site.getsitepackages():
        for relative_path in _CUDA_DLL_DIRECTORIES:
            directory = os.path.join(site_packages, *relative_path)
            if os.path.isdir(directory):
                try:
                    _DLL_HANDLES.append(os.add_dll_directory(directory))
                    logger.debug(f"Registered DLL directory: {directory}")
                except OSError:
                    pass

    # 2. Preload CUDA DLLs to resolve runtime dependencies
    for dll_name in _PRELOAD_DLLS:
        if dll_name not in _LOADED_DLL_HANDLES:
            try:
                # Load with WinDLL to register them globally in the process address space
                handle = ctypes.WinDLL(dll_name)
                _LOADED_DLL_HANDLES[dll_name] = handle
                logger.debug(f"Successfully preloaded DLL: {dll_name}")
            except Exception as e:
                logger.warning(f"Failed to preload DLL {dll_name}: {e}")
