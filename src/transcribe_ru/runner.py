"""Общая оркестрация транскрипции — единая точка для CLI и GUI.

Собирает связку выбор устройства → движок → нарезка/распознавание → форматирование
→ запись файла. И `cli`, и `gui` ходят сюда, чтобы логика не дублировалась.
"""

from __future__ import annotations

import gc
from pathlib import Path

from transcribe_ru.formatters import EXTENSIONS, format_result


def _empty_cache(device: str) -> None:
    """Освободить кэш ускорителя между файлами пакета (без жёсткой зависимости от torch)."""
    try:
        import torch

        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
    except Exception:  # torch может отсутствовать (тесты) — не критично
        pass


def output_path(audio_path: str, fmt: str, out_dir: str | None) -> Path:
    """Путь результата: имя исходника + расширение формата."""
    src = Path(audio_path)
    name = src.stem + EXTENSIONS[fmt]
    folder = Path(out_dir) if out_dir else src.parent
    return folder / name


def _write_result(result, fmt: str, meta: dict, out_file: Path) -> None:
    """Отформатировать сегменты и АТОМАРНО записать в `out_file` (temp + rename).

    Атомарность важна для пакета с resume: оборванный на полпути файл не должен
    остаться как «готовый» результат, который следующий прогон пропустит.
    """
    text = format_result(result.segments, fmt, meta=meta)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.with_suffix(out_file.suffix + ".part")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(out_file)


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

    out_file = output_path(audio_path, fmt, out_dir)
    _write_result(result, fmt, {"engine": engine, "variant": variant}, out_file)
    return out_file


def transcribe_batch(
    paths,
    *,
    engine: str = "gigaam",
    variant: str = "e2e_rnnt",
    device: str = "auto",
    fmt: str = "txt_timecoded",
    granularity: str = "coarse",
    out_dir: str | None = None,
    skip_existing: bool = False,
    progress_factory=None,
    on_done=None,
    on_error=None,
    on_skip=None,
    # точки инъекции для тестов:
    transcribe_fn=None,
    get_engine_fn=None,
    select_device_fn=None,
) -> dict:
    """Транскрибировать список файлов по очереди, загрузив движок ОДИН раз.

    Общая петля для CLI (папка) и GUI (мультивыбор): резолвит устройство и грузит
    модель один раз, затем идёт по `paths`, пишет результат рядом с каждым
    исходником (или в `out_dir`). Один битый файл не роняет пакет — ошибка
    копится в сводке. `skip_existing` пропускает файлы с уже готовым результатом
    (resume для папок).

    Колбэки (все опциональны, получают `path` — исходный элемент из `paths`):
    - `progress_factory(idx, total, path) -> on_progress(done, total) | None` —
      вызывается в начале каждого файла (тут же можно анонсировать старт),
      возвращает колбэк прогресса по окнам этого файла;
    - `on_done(idx, total, path, out_file, n_seg)` — файл готов;
    - `on_error(idx, total, path, error)` — файл упал;
    - `on_skip(idx, total, path)` — файл пропущен (skip_existing).

    Возвращает сводку `{"done": [Path], "skipped": [str], "failed": [(str, str)]}`.
    """
    if transcribe_fn is None:
        from transcribe_ru.core import transcribe as transcribe_fn
    if get_engine_fn is None:
        from transcribe_ru.engines.base import get_engine as get_engine_fn
    if select_device_fn is None:
        from transcribe_ru.device import select_device as select_device_fn

    resolved_device = select_device_fn(device)
    engine_obj = get_engine_fn(engine)(variant=variant)
    engine_obj.load(resolved_device)         # модель грузится ОДИН раз на весь пакет
    engine_obj.load = lambda _device: None   # дальше core.transcribe её не перегружает

    meta = {"engine": engine, "variant": variant}
    total = len(paths)
    done: list[Path] = []
    skipped: list[str] = []
    failed: list[tuple[str, str]] = []

    for idx, path in enumerate(paths, start=1):
        name = Path(path).name
        out_file = output_path(path, fmt, out_dir)

        if skip_existing and out_file.exists() and out_file.stat().st_size > 0:
            skipped.append(name)
            if on_skip is not None:
                on_skip(idx, total, path)
            continue

        on_progress = progress_factory(idx, total, path) if progress_factory else None
        try:
            result = transcribe_fn(
                str(path), engine_obj, device=resolved_device,
                granularity=granularity, on_progress=on_progress,
            )
            _write_result(result, fmt, meta, out_file)
            done.append(out_file)
            if on_done is not None:
                on_done(idx, total, path, out_file, len(result.segments))
        except Exception as exc:  # один битый файл не должен ронять весь пакет
            failed.append((name, f"{type(exc).__name__}: {exc}"))
            if on_error is not None:
                on_error(idx, total, path, exc)
        finally:
            gc.collect()
            _empty_cache(resolved_device)

    return {"done": done, "skipped": skipped, "failed": failed}
