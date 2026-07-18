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
import gc
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
    from transcribe_ru.core import transcribe
    from transcribe_ru.device import select_device
    from transcribe_ru.engines.gigaam import GigaAMEngine
    from transcribe_ru.formatters import EXTENSIONS, format_result

    configure_ssl()
    ext = EXTENSIONS[args.format]

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

    device = select_device(args.device)
    log(f"Устройство: {device}; движок: gigaam/{args.variant}; "
        f"формат: {args.format}; детализация: {args.granularity}")
    log(f"Найдено аудиофайлов: {len(files)}")

    log("Загрузка модели (один раз)...")
    t0 = time.monotonic()
    engine = GigaAMEngine(variant=args.variant)
    engine.load(device)
    # Не перегружать модель на каждом файле: core.transcribe зовёт load() внутри.
    engine.load = lambda _device: None
    log(f"Модель загружена за {time.monotonic() - t0:.1f}с")

    meta = {"engine": "gigaam", "variant": args.variant}
    done = skipped = 0
    failed: list[tuple[str, str]] = []
    run_start = time.monotonic()

    for i, src in enumerate(files, 1):
        out_file = src.with_suffix(ext)
        size_mb = src.stat().st_size / 1024 / 1024

        if out_file.exists() and out_file.stat().st_size > 0:
            log(f"[{i}/{len(files)}] ПРОПУСК (уже есть): {src.name}")
            skipped += 1
            continue

        log(f"[{i}/{len(files)}] СТАРТ: {src.name} ({size_mb:.0f} МБ)")
        t_file = time.monotonic()
        try:
            result = transcribe(
                str(src), engine, device=device, granularity=args.granularity,
                on_progress=_make_progress(src.name[:40]),
            )
            text = format_result(result.segments, args.format, meta=meta)
            # Атомарная запись: сперва во временный файл, затем rename.
            tmp = out_file.with_suffix(out_file.suffix + ".part")
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(out_file)

            dt = time.monotonic() - t_file
            n_seg = len(result.segments)
            log(
                f"[{i}/{len(files)}] ГОТОВО: {src.name} → {out_file.name} "
                f"({n_seg} сегм., {_fmt_dur(dt)} обработки)"
            )
            done += 1
            del result, text
        except Exception as exc:  # один битый файл не должен ронять весь прогон
            log(f"[{i}/{len(files)}] ОШИБКА: {src.name}: {type(exc).__name__}: {exc}")
            failed.append((src.name, f"{type(exc).__name__}: {exc}"))
        finally:
            gc.collect()
            try:
                import torch

                if device == "mps":
                    torch.mps.empty_cache()
                elif device == "cuda":
                    torch.cuda.empty_cache()
            except Exception:
                pass

    total_dt = time.monotonic() - run_start
    log("=" * 60)
    log(f"ИТОГО: обработано {done}, пропущено {skipped}, ошибок {len(failed)} "
        f"за {_fmt_dur(total_dt)}")
    for name, err in failed:
        log(f"  СБОЙ: {name}: {err}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
