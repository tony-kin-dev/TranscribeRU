"""Загрузка аудио любого формата через системный ffmpeg → 16 kHz mono float32."""

from __future__ import annotations

import subprocess

import numpy as np

SAMPLE_RATE = 16000


def load(path: str, *, ffmpeg_bin: str = "ffmpeg", sample_rate: int = SAMPLE_RATE):
    """Декодировать аудиофайл в `(float32 ndarray, sample_rate)`.

    Любой вход (`.opus/.m4a/.wav/.mp3/…`) приводится к PCM s16le, моно,
    `sample_rate` Гц, затем нормализуется в float32 в диапазоне [-1, 1].
    """
    cmd = [
        ffmpeg_bin, "-nostdin", "-threads", "0",
        "-i", path,
        "-f", "s16le", "-ac", "1", "-ar", str(sample_rate),
        "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"ffmpeg не найден ({ffmpeg_bin!r}). Установите его: "
            "macOS — `brew install ffmpeg`, Debian/Ubuntu — `apt install ffmpeg`, "
            "Windows — `winget install ffmpeg`."
        ) from exc

    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"ffmpeg не смог декодировать {path!r}:\n{stderr}")

    pcm = np.frombuffer(proc.stdout, dtype=np.int16)
    return (pcm.astype(np.float32) / 32768.0), sample_rate
