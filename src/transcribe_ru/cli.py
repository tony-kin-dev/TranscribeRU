"""CLI: разбор аргументов, прогресс в stderr, запись результата рядом с исходником."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from transcribe_ru.engines.gigaam import DEFAULT_VARIANT
from transcribe_ru.formatters import EXTENSIONS, format_result

FORMATS = ("txt_timecoded", "txt_plain", "srt", "json")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="transcribe-ru",
        description="Распознавание речи (ASR) на PyTorch-движках; движок по умолчанию — GigaAM-v3.",
    )
    p.add_argument("--audio", required=True, help="Входной аудиофайл")
    p.add_argument("--engine", default="gigaam", help="Движок ASR (по умолчанию gigaam)")
    p.add_argument("--variant", default=DEFAULT_VARIANT, help="Вариант модели движка")
    p.add_argument(
        "--device", default="auto", choices=("auto", "cuda", "mps", "cpu"),
        help="Устройство вычислений",
    )
    p.add_argument(
        "--format", default="txt_timecoded", choices=FORMATS, help="Формат вывода",
    )
    p.add_argument("--out-dir", default=None, help="Каталог результата (по умолчанию — рядом с исходником)")
    p.add_argument("--dry-run", action="store_true", help="Показать план и выйти")
    return p


def output_path(audio_path: str, fmt: str, out_dir: str | None) -> Path:
    """Путь результата: имя исходника + расширение формата."""
    src = Path(audio_path)
    name = src.stem + EXTENSIONS[fmt]
    folder = Path(out_dir) if out_dir else src.parent
    return folder / name


def _make_progress(stream):
    """Колбэк прогресса: готово/всего, %, ETA — в поток (stderr)."""
    start = time.monotonic()

    def on_progress(done: int, total: int) -> None:
        pct = 100.0 * done / total if total else 100.0
        elapsed = time.monotonic() - start
        eta = (elapsed / done) * (total - done) if done else 0.0
        end = "\n" if done >= total else ""
        stream.write(f"\r{done}/{total} ({pct:.0f}%) ETA {eta:5.1f}s{end}")
        stream.flush()

    return on_progress


def main(
    argv=None,
    *,
    transcribe_fn=None,
    get_engine_fn=None,
    select_device_fn=None,
    stderr=None,
) -> int:
    args = build_parser().parse_args(argv)
    stderr = stderr or sys.stderr

    if transcribe_fn is None:
        from transcribe_ru.core import transcribe as transcribe_fn
    if get_engine_fn is None:
        from transcribe_ru.engines.base import get_engine as get_engine_fn
    if select_device_fn is None:
        from transcribe_ru.device import select_device as select_device_fn

    device = select_device_fn(args.device)
    out_file = output_path(args.audio, args.format, args.out_dir)

    if args.dry_run:
        stderr.write(
            f"[dry-run] движок={args.engine} вариант={args.variant} устройство={device}\n"
            f"[dry-run] {args.audio} → {out_file} (формат {args.format})\n"
        )
        return 0

    engine_cls = get_engine_fn(args.engine)
    engine = engine_cls(variant=args.variant)

    result = transcribe_fn(
        args.audio, engine, device=device, on_progress=_make_progress(stderr)
    )

    meta = {"engine": args.engine, "variant": args.variant}
    text = format_result(result.segments, args.format, meta=meta)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(text, encoding="utf-8")
    stderr.write(f"Готово: {out_file}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
