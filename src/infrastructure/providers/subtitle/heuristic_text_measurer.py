from src.application.ports.subtitle_text_measurer import SubtitleTextMeasurer

class HeuristicTextMeasurer:
    """Deterministic measurer that approximates text width without reading font files.

    Width model:
        - wide chars (CJK, fullwidth, hangul): font_size * 1.0
        - spaces: font_size * 0.3
        - other letters/digits: font_size * 0.55
        - bold: +5 percent
        - spacing: spacing * (n - 1) added to width

    Height is reported as font_size for a single line.
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
        if font_size <= 0:
            raise ValueError("font_size must be positive.")
        if not text:
            return 0.0, float(font_size)
        width = 0.0
        for char in text:
            if char.isspace():
                width += font_size * 0.3
            elif self._is_wide(char):
                width += font_size * 1.0
            else:
                width += font_size * 0.55
        if len(text) > 1:
            width += spacing * (len(text) - 1)
        if bold:
            width *= 1.05
        return width, float(font_size)

    @staticmethod
    def _is_wide(char: str) -> bool:
        cp = ord(char)
        return (
            0x3000 <= cp <= 0x9FFF
            or 0xFF00 <= cp <= 0xFFEF
            or 0x3040 <= cp <= 0x30FF
            or 0xAC00 <= cp <= 0xD7AF
        )
