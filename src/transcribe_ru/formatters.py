"""Форматтеры: список сегментов → строка нужного формата.

Сегмент — любой объект с атрибутами `start`/`end` (секунды, float) и `text`.
"""

from __future__ import annotations

import json

EXTENSIONS = {
    "txt_timecoded": ".txt",
    "txt_plain": ".txt",
    "srt": ".srt",
    "json": ".json",
}


def _hms(seconds: float) -> str:
    """Секунды → `HH:MM:SS` (без долей)."""
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _srt_time(seconds: float) -> str:
    """Секунды → `HH:MM:SS,mmm` (формат SRT)."""
    ms_total = round(seconds * 1000)
    ms = ms_total % 1000
    total = ms_total // 1000
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_txt_timecoded(segments) -> str:
    blocks = [f"{_hms(seg.start)}\n{seg.text}" for seg in segments]
    return "\n\n".join(blocks) + "\n" if blocks else ""


def to_txt_plain(segments) -> str:
    return " ".join(seg.text for seg in segments)


def to_srt(segments) -> str:
    blocks = []
    for i, seg in enumerate(segments, start=1):
        blocks.append(
            f"{i}\n{_srt_time(seg.start)} --> {_srt_time(seg.end)}\n{seg.text}"
        )
    return "\n\n".join(blocks) + "\n" if blocks else ""


def to_json(segments, meta=None) -> str:
    payload = dict(meta or {})
    payload["segments"] = [
        {"start": seg.start, "end": seg.end, "text": seg.text} for seg in segments
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def format_result(segments, fmt: str, *, meta=None) -> str:
    """Отформатировать сегменты в строку формата `fmt`."""
    if fmt == "txt_timecoded":
        return to_txt_timecoded(segments)
    if fmt == "txt_plain":
        return to_txt_plain(segments)
    if fmt == "srt":
        return to_srt(segments)
    if fmt == "json":
        return to_json(segments, meta)
    raise ValueError(
        f"Неизвестный формат {fmt!r}; допустимы: {', '.join(EXTENSIONS)}"
    )
