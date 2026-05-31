"""CLI: разбор аргументов, прогресс в stderr, запись результата рядом с исходником."""

from __future__ import annotations

import argparse
import sys
import time

from transcribe_ru.certs import configure_ssl
from transcribe_ru.engines.gigaam import DEFAULT_VARIANT
from transcribe_ru.runner import output_path, transcribe_to_file

FORMATS = ("txt_timecoded", "txt_plain", "srt", "json")

# Совместимость: output_path исторически жил в cli; теперь общий, в runner.
__all__ = ["build_parser", "output_path", "main"]


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
    configure_ssl()  # лечит CERTIFICATE_VERIFY_FAILED при скачивании модели
    args = build_parser().parse_args(argv)
    stderr = stderr or sys.stderr

    if args.dry_run:
        # для плана достаточно резолва устройства и пути; саму работу не делаем
        if select_device_fn is None:
            from transcribe_ru.device import select_device as select_device_fn
        device = select_device_fn(args.device)
        out_file = output_path(args.audio, args.format, args.out_dir)
        stderr.write(
            f"[dry-run] движок={args.engine} вариант={args.variant} устройство={device}\n"
            f"[dry-run] {args.audio} → {out_file} (формат {args.format})\n"
        )
        return 0

    out_file = transcribe_to_file(
        args.audio,
        engine=args.engine,
        variant=args.variant,
        device=args.device,
        fmt=args.format,
        out_dir=args.out_dir,
        on_progress=_make_progress(stderr),
        transcribe_fn=transcribe_fn,
        get_engine_fn=get_engine_fn,
        select_device_fn=select_device_fn,
    )
    stderr.write(f"Готово: {out_file}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
