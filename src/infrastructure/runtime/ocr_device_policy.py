from dataclasses import dataclass

from src.shared.exceptions import ProviderError


@dataclass(frozen=True)
class OCRDeviceStatus:
    requested: str
    used: str
    paddle_cuda_available: bool
    paddle_device: str


def validate_ocr_device(requested: str) -> OCRDeviceStatus:
    if requested == "cpu":
        return OCRDeviceStatus("cpu", "cpu", False, "cpu")
    try:
        import paddle

        cuda_available = bool(paddle.device.is_compiled_with_cuda())
        paddle_device = str(paddle.device.get_device())
    except Exception as error:
        raise ProviderError("OCR_DEVICE_GPU_REQUIRED_BUT_UNAVAILABLE") from error
    if not cuda_available or not paddle_device.lower().startswith("gpu"):
        raise ProviderError("OCR_DEVICE_GPU_REQUIRED_BUT_UNAVAILABLE")
    return OCRDeviceStatus(requested, "gpu", True, paddle_device)
