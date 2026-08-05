from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SubtitleEvent:
    start_ms: int
    end_ms: int
    text: str
    speaker_id: str = "SPK_00"
    color: str = "#FFFFFF"
    layer: int = 0


def _ass_time(milliseconds: int) -> str:
    total_centiseconds = max(0, milliseconds) // 10
    hours, remainder = divmod(total_centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    seconds, centiseconds = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def _ass_color(rgb: str) -> str:
    value = rgb.removeprefix("#").upper()
    if len(value) != 6:
        value = "FFFFFF"
    return f"&H00{value[4:6]}{value[2:4]}{value[0:2]}"


def _escape_text(value: str) -> str:
    line_break = "\0KAOR_LINE_BREAK\0"
    return (
        value.replace(r"\N", line_break)
        .replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\r\n", r"\N")
        .replace("\n", r"\N")
        .replace(line_break, r"\N")
    )


def build_ass(
    events: Iterable[SubtitleEvent],
    *,
    width: int,
    height: int,
    font_name: str = "Noto Sans SC",
    font_size: int = 48,
    margin_vertical: int = 56,
    outline: float = 2.4,
    target_region: tuple[float, float, float, float] | None = None,
) -> str:
    rows = list(events)
    layers = sorted({event.layer for event in rows}) or [0]
    layer_rank = {layer: index for index, layer in enumerate(layers)}
    effective_font_size = font_size
    region_top = region_height_px = 0
    if target_region is not None:
        _, y, _, region_height = target_region
        region_top = round(y * height)
        region_height_px = max(1, round(region_height * height))
        layer_span = 1 + max(0, len(layers) - 1) * 1.35
        fitting_size = max(6, int(region_height_px / layer_span))
        effective_font_size = min(font_size, fitting_size)
    line_step = round(effective_font_size * 1.35)
    stacked_height = effective_font_size + max(0, len(layers) - 1) * line_step
    first_line_y = region_top + max(0, round((region_height_px - stacked_height) / 2))
    styles: dict[tuple[str, str, int], str] = {}
    for event in rows:
        key = (event.speaker_id, event.color.upper(), event.layer)
        styles.setdefault(key, f"Speaker_{len(styles) + 1:02d}")
    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "ScaledBorderAndShadow: yes",
        "WrapStyle: 0",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
    ]
    for (speaker_id, color, layer), style_name in styles.items():
        vertical_margin = margin_vertical + layer_rank[layer] * line_step
        header.append(
            "Style: "
            f"{style_name},{font_name},{effective_font_size},{_ass_color(color)},&H00FFFFFF,"
            f"&H00101010,&H70000000,0,0,0,0,100,100,0,0,1,{outline},0.8,2,"
            f"36,36,{vertical_margin},1"
        )
    header.extend(
        [
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]
    )
    for event in sorted(rows, key=lambda item: (item.start_ms, item.layer)):
        style_name = styles[(event.speaker_id, event.color.upper(), event.layer)]
        text = _escape_text(event.text)
        if target_region is not None:
            x, y, region_width, region_height = target_region
            anchor_x = round((x + region_width / 2) * width)
            anchor_y = first_line_y + layer_rank[event.layer] * line_step
            maximum_y = round((y + region_height) * height - effective_font_size)
            anchor_y = min(anchor_y, maximum_y)
            text = rf"{{\an8\pos({anchor_x},{anchor_y})}}{text}"
        header.append(
            f"Dialogue: {event.layer},{_ass_time(event.start_ms)},{_ass_time(event.end_ms)},"
            f"{style_name},{event.speaker_id},0,0,0,,{text}"
        )
    return "\n".join(header) + "\n"


def write_ass(path: Path, events: Iterable[SubtitleEvent], **options: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_ass(events, **options), encoding="utf-8-sig")
    return path
