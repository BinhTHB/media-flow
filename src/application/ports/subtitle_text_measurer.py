from typing import Protocol, runtime_checkable

@runtime_checkable
class SubtitleTextMeasurer(Protocol):
    """Application port for measuring subtitle text dimensions.

    Implementations must be deterministic and side-effect free.
    No filesystem, network, or font file reading is required at this layer.
    """

    def measure(
        self,
        text: str,
        *,
        font_family: str,
        font_size: int,
        bold: bool,
        spacing: float,
    ) -> tuple[float, float]:
        """Return (width_px, height_px) for a single line of text."""
