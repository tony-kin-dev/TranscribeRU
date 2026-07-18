"""Общая оркестрация транскрипции — единая точка для CLI и GUI.

Собирает связку выбор устройства → движок → нарезка/распознавание → форматирование
→ запись файла. И `cli`, и `gui` ходят сюда, чтобы логика не дублировалась.
"""

from __future__ import annotations

from pathlib import Path

from transcribe_ru.formatters import EXTENSIONS, format_result


def output_path(audio_path: str, fmt: str, out_dir: str | None) -> Path:
    """Путь результата: имя исходника + расширение формата."""
    src = Path(audio_path)
    name = src.stem + EXTENSIONS[fmt]
    folder = Path(out_dir) if out_dir else src.parent
    return folder / name


def transcribe_to_file(
    audio_path,
    *,
    engine: str = "gigaam",
    variant: str = "e2e_rnnt",
    device: str = "auto",
    fmt: str = "txt_timecoded",
    granularity: str = "coarse",
    out_dir: str | None = None,
    on_progress=None,
    # точки инъекции для тестов:
    transcribe_fn=None,
    get_engine_fn=None,
    select_device_fn=None,
) -> Path:
    """Транскрибировать файл и записать результат; вернуть путь к файлу."""
    if transcribe_fn is None:
        from transcribe_ru.core import transcribe as transcribe_fn
    if get_engine_fn is None:
        from transcribe_ru.engines.base import get_engine as get_engine_fn
    if select_device_fn is None:
        from transcribe_ru.device import select_device as select_device_fn

    resolved_device = select_device_fn(device)
    engine_obj = get_engine_fn(engine)(variant=variant)

    result = transcribe_fn(
        audio_path, engine_obj, device=resolved_device,
        granularity=granularity, on_progress=on_progress,
    )

    meta = {"engine": engine, "variant": variant}
    text = format_result(result.segments, fmt, meta=meta)

    out_file = output_path(audio_path, fmt, out_dir)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(text, encoding="utf-8")
    return out_file
