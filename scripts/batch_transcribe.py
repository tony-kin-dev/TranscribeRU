#!/usr/bin/env python
"""Пакетная транскрипция всех аудиофайлов в папке одним проходом.

Грузит движок GigaAM-v3 ОДИН раз и переиспользует между файлами (экономит
повторную загрузку модели). Поддерживает resume (пропуск уже готовых),
устойчив к ошибкам отдельного файла, чистит память между файлами и пишет
подробный лог в stdout (перенаправляйте в файл при фоновом запуске).

Использование:
    python scripts/batch_transcribe.py --src <папка> [--format txt_timecoded]
                                       [--variant e2e_rnnt] [--granularity coarse]
                                       [--limit N] [--reverse]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

AUDIO_EXTS = {".m4a", ".mp3", ".wav", ".mp4", ".mov", ".aac", ".opus", ".flac", ".ogg"}


def log(msg: str) -> None:
    """Печать с временной меткой и немедленным сбросом буфера (для tail -f)."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _fmt_dur(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def _make_progress(label: str):
    """Прогресс по окнам внутри файла; печать не чаще раза в 15 с."""
    start = time.monotonic()
    last = [0.0]

    def on_progress(done: int, total: int) -> None:
        now = time.monotonic()
        if done < total and (now - last[0]) < 15.0:
            return
        last[0] = now
        pct = 100.0 * done / total if total else 100.0
        elapsed = now - start
        eta = (elapsed / done) * (total - done) if done else 0.0
        end = "\n" if done >= total else ""
        sys.stdout.write(
            f"\r    {label}: окно {done}/{total} ({pct:.0f}%) "
            f"прошло {elapsed:5.0f}с ETA {eta:5.0f}с{end}"
        )
        sys.stdout.flush()

    return on_progress


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Пакетная транскрипция папки")
    p.add_argument("--src", required=True, help="Папка с аудиофайлами")
    p.add_argument("--format", default="txt_timecoded")
    p.add_argument("--variant", default="e2e_rnnt")
    p.add_argument("--granularity", default="coarse", choices=("coarse", "fine"),
                   help="coarse — крупные окна ~20-24с; fine — короткие реплики ~5-7с (субтитры)")
    p.add_argument("--device", default="auto")
    p.add_argument("--limit", type=int, default=None, help="Обработать только первые N (для теста)")
    p.add_argument("--reverse", action="store_true", help="Сначала самые большие")
    args = p.parse_args(argv)

    src_dir = Path(args.src)
    if not src_dir.is_dir():
        log(f"ОШИБКА: папка не найдена: {src_dir}")
        return 2

    # Импорты проекта откладываем, чтобы --help был мгновенным.
    from transcribe_ru.certs import configure_ssl
    from transcribe_ru.runner import transcribe_batch

    configure_ssl()

    # Собираем файлы, сортируем по размеру (короткие первыми — раньше виден результат).
    files = sorted(
        (f for f in src_dir.iterdir() if f.suffix.lower() in AUDIO_EXTS),
        key=lambda f: f.stat().st_size,
        reverse=args.reverse,
    )
    if args.limit:
        files = files[: args.limit]

    if not files:
        log(f"В папке нет аудиофайлов: {src_dir}")
        return 1

    log(f"Движок: gigaam/{args.variant}; формат: {args.format}; "
        f"детализация: {args.granularity}; устройство: {args.device}")
    log(f"Найдено аудиофайлов: {len(files)}; загрузка модели (один раз)…")

    def progress_factory(idx, total, path):
        p = Path(path)
        size_mb = p.stat().st_size / 1024 / 1024
        log(f"[{idx}/{total}] СТАРТ: {p.name} ({size_mb:.0f} МБ)")
        return _make_progress(p.name[:40])

    def on_done(idx, total, path, out_file, n_seg):
        log(f"[{idx}/{total}] ГОТОВО: {Path(path).name} → {out_file.name} ({n_seg} сегм.)")

    def on_error(idx, total, path, err):
        log(f"[{idx}/{total}] ОШИБКА: {Path(path).name}: {type(err).__name__}: {err}")

    def on_skip(idx, total, path):
        log(f"[{idx}/{total}] ПРОПУСК (уже есть): {Path(path).name}")

    run_start = time.monotonic()
    summary = transcribe_batch(
        [str(f) for f in files],
        variant=args.variant,
        device=args.device,
        fmt=args.format,
        granularity=args.granularity,
        skip_existing=True,
        progress_factory=progress_factory,
        on_done=on_done,
        on_error=on_error,
        on_skip=on_skip,
    )
    total_dt = time.monotonic() - run_start

    log("=" * 60)
    log(f"ИТОГО: обработано {len(summary['done'])}, пропущено {len(summary['skipped'])}, "
        f"ошибок {len(summary['failed'])} за {_fmt_dur(total_dt)}")
    for name, err in summary["failed"]:
        log(f"  СБОЙ: {name}: {err}")
    return 0 if not summary["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
