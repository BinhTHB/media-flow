from dataclasses import dataclass

from src.shared.exceptions import ProviderError


@dataclass(frozen=True)
class OCRDeviceStatus:
    requested: str
    used: str
    cuda_available: bool
    gpu_device: str


def validate_ocr_device(requested: str) -> OCRDeviceStatus:
    if requested == "cpu":
        return OCRDeviceStatus("cpu", "cpu", False, "cpu")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except Exception as error:
        raise ProviderError("OCR_DEVICE_GPU_REQUIRED_BUT_UNAVAILABLE") from error
    if not cuda_available:
        raise ProviderError("OCR_DEVICE_GPU_REQUIRED_BUT_UNAVAILABLE")
    return OCRDeviceStatus(requested, "gpu", True, "gpu")
