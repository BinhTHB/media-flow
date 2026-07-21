import logging
from typing import List, Optional
from src.domain.entities.segment import Segment
from src.domain.entities.subtitle_placement import SubtitlePlacement
from src.domain.entities.subtitle_style import SubtitleStyle

def _format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centiseconds = int(round((seconds - int(seconds)) * 100))
    if centiseconds == 100:
        secs += 1
        centiseconds = 0
    return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"

def _color_to_ass(color: str, alpha: float = 1.0) -> str:
    if color.startswith("&H"):
        return color
    rrggbb = color.lstrip("#")
    if len(rrggbb) != 6:
        rrggbb = "FFFFFF"
    r = rrggbb[0:2]
    g = rrggbb[2:4]
    b = rrggbb[4:6]
    a_val = int(round((1 - alpha) * 255))
    return f"&H{a_val:02X}{b}{g}{r}"

def _escape_text(text: str) -> str:
    return text.replace("{", r"\{").replace("}", r"\}").replace("\n", r"\N")

def _border_style(background: str) -> int:
    if background == "box":
        return 3
    return 1

def generate_ass_header(
    style: SubtitleStyle, play_res_x: int, play_res_y: int
) -> str:
    primary = _color_to_ass(style.primary_color, style.opacity)
    secondary = _color_to_ass("#000000")  # karaoke
    outline = _color_to_ass(style.outline_color)
    back = _color_to_ass("#000000", 1)  # background
    bold_val = "-1" if style.bold else "0"
    italic_val = "-1" if style.italic else "0"
    border_style_val = _border_style(style.background)

    lines = []
    lines.append("[Script Info]")
    lines.append("ScriptType: v4.00+")
    lines.append(f"PlayResX: {play_res_x}")
    lines.append(f"PlayResY: {play_res_y}")
    lines.append("ScaledBorderAndShadow: yes")
    lines.append("")
    lines.append("[V4+ Styles]")
    lines.append("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding")
    lines.append(
        f"Style: Default,{style.font_family},{style.font_size},"
        f"{primary},{secondary},{outline},{back},"
        f"{bold_val},{italic_val},0,0,100,{style.line_height * 100:.0f},"
        f"{style.spacing},0,{border_style_val},{style.outline_width},"
        f"{style.shadow},{style.alignment},{style.margin_l},"
        f"{style.margin_r},{style.margin_v},1"
    )
    lines.append("")
    lines.append("[Events]")
    lines.append("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text")
    return "\n".join(lines)

def _has_mask_provenance(region: "SubtitleRegion") -> bool:
    """A region is eligible for production masking only when it carries classifier provenance."""
    return bool(getattr(region, "classification_id", None)) and bool(getattr(region, "is_subtitle", None)) and bool(getattr(region, "classifier_model", None))


class ASSGenerator:
    @staticmethod
    def generate(
        segments: List[Segment],
        style: SubtitleStyle,
        placements: Optional[List[SubtitlePlacement]] = None,
        play_res_x: int = 1920,
        play_res_y: int = 1080,
        mask_regions: Optional[List["SubtitleRegion"]] = None,
    ) -> str:
        if not segments:
            return ""
        header = generate_ass_header(style, play_res_x, play_res_y)
        events = []

        if mask_regions:
            for r in mask_regions:
                # Gate: production mask regions must carry explicit confirmation provenance.
                if not _has_mask_provenance(r):
                    logging.getLogger(__name__).warning(
                        "LOUD-WARN [ASSGenerator]: discarding mask_region without verified classifier provenance. "
                        "Region details: start=%.3f, end=%.3f, x=%.3f, y=%.3f",
                        r.start_time, r.end_time, r.x, r.y
                    )
                    continue

                # Che đen đặc, padding 8px
                pad = 8
                left = max(0.0, r.x * play_res_x - pad)
                top = max(0.0, r.y * play_res_y - pad)
                right = min(float(play_res_x), (r.x + r.width) * play_res_x + pad)
                bottom = min(float(play_res_y), (r.y + r.height) * play_res_y + pad)
                width = round(right - left)
                height = round(bottom - top)
                
                # Nới rộng ±2 frame (khoảng 0.08s) để mượt
                r_start = _format_time(max(0.0, r.start_time - 0.08))
                r_end = _format_time(r.end_time + 0.08)
                
                events.append(
                    f"Dialogue: 0,{r_start},{r_end},Default,,0,0,0,,"
                    rf"{{\an7\pos({round(left)},{round(top)})"
                    rf"\p1\1c&H000000&\1a&H00&}}"
                    f"m 0 0 l {width} 0 l {width} {height} l 0 {height}"
                    rf"{{\p0}}{{provenance={r.classification_id}}}"
                )

        for index, seg in enumerate(segments):
            if seg.translated_text:
                text = seg.translated_text
            else:
                logging.getLogger(__name__).warning(
                    "LOUD-WARN [ASSGenerator]: segment %s has EMPTY translated_text; "
                    "falling back to source text %r. Translation did not run or failed.",
                    seg.id,
                    seg.text[:40],
                )
                text = seg.text
            start = _format_time(seg.start)
            end = _format_time(seg.end)
            escaped = _escape_text(text)
            margin_l = margin_r = margin_v = "0"
            tags = []
            layer = 0
            if placements is not None and index < len(placements):
                placement = placements[index]
                if placement is not None:
                    # Nếu dùng mask_regions thì bỏ qua sinh mask_bbox ở Layer 0
                    if placement.mask_bbox is not None and not mask_regions:
                        mask = placement.mask_bbox
                        width = round(mask["right"] - mask["left"])
                        height = round(mask["bottom"] - mask["top"])
                        events.append(
                            f"Dialogue: 0,{start},{end},Default,,0,0,0,,"
                            rf"{{\an7\pos({round(mask['left'])},{round(mask['top'])})"
                            rf"\p1\1c&H000000&\1a&H00&}}"
                            f"m 0 0 l {width} 0 l {width} {height} l 0 {height}"
                            r"{\p0}"
                        )
                        layer = 1
                    elif mask_regions and seg.source == "ocr":
                        # Có mask_regions: text layer cho OCR là 1 để hiển thị đè lên mask
                        layer = 1
                    if placement.margin_l is not None:
                        margin_l = str(placement.margin_l)
                    if placement.margin_r is not None:
                        margin_r = str(placement.margin_r)
                    if placement.margin_v is not None:
                        margin_v = str(placement.margin_v)
                    if placement.alignment != style.alignment:
                        tags.append(rf"\an{placement.alignment}")
                    if placement.x is not None and placement.y is not None:
                        tags.append(rf"\pos({placement.x},{placement.y})")
            if tags:
                escaped = f"{{{''.join(tags)}}}{escaped}"
            events.append(
                f"Dialogue: {layer},{start},{end},Default,,{margin_l},{margin_r},{margin_v},,{escaped}"
            )
        return header + "\n" + "\n".join(events)
