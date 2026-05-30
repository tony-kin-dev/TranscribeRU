"""Нарезка длинного аудио на окна <25с по границам речи (silero-vad).

`silero-vad` находит интервалы речи (MIT, без HF-токена). Ядро склеивает
соседние короткие интервалы до целевой длины и режет слишком длинные так,
чтобы каждое окно было ниже порога longform GigaAM. Границы окон становятся
тайм-кодами сегментов на выходе.
"""

from __future__ import annotations

import math

# По умолчанию ниже LONGFORM_THRESHOLD GigaAM (~25с) с запасом.
MAX_LEN = 24.0
TARGET_LEN = 20.0


def pack_windows(intervals, *, max_len=MAX_LEN, target_len=TARGET_LEN):
    """Речевые интервалы (сек) → окна, каждое не длиннее `max_len`.

    Длинные интервалы режутся на равные части ниже `max_len`; соседние
    интервалы жадно сливаются, пока суммарная длина окна <= `target_len`.
    """
    parts: list[tuple[float, float]] = []
    for start, end in intervals:
        dur = end - start
        if dur <= max_len:
            parts.append((start, end))
            continue
        n = math.ceil(dur / max_len)
        step = dur / n
        for i in range(n):
            piece_end = end if i == n - 1 else start + (i + 1) * step
            parts.append((start + i * step, piece_end))

    windows: list[tuple[float, float]] = []
    current: list[float] | None = None
    for start, end in parts:
        if current is None:
            current = [start, end]
        elif (end - current[0]) <= target_len:
            current[1] = end
        else:
            windows.append((current[0], current[1]))
            current = [start, end]
    if current is not None:
        windows.append((current[0], current[1]))
    return windows


def slice_audio(wav, sr, windows):
    """Нарезать сигнал на куски по окнам; вернуть [(chunk, (start, end)), …]."""
    chunks = []
    for start, end in windows:
        i0 = int(round(start * sr))
        i1 = int(round(end * sr))
        chunks.append((wav[i0:i1], (start, end)))
    return chunks


def _silero_speech_timestamps(wav, sr):
    """Интервалы речи (сек) через silero-vad."""
    from silero_vad import get_speech_timestamps, load_silero_vad

    model = load_silero_vad()
    stamps = get_speech_timestamps(
        wav, model, sampling_rate=sr, return_seconds=True
    )
    return [(s["start"], s["end"]) for s in stamps]


def segment(wav, sr, *, max_len=MAX_LEN, target_len=TARGET_LEN, get_speech=None):
    """Аудио → список окон <`max_len` с границами.

    `get_speech(wav, sr) -> [(start_sec, end_sec), …]` инъектируется в тестах;
    по умолчанию используется silero-vad. Если речь не найдена — весь файл
    обрабатывается как один интервал.
    """
    if get_speech is None:
        get_speech = _silero_speech_timestamps

    intervals = get_speech(wav, sr)
    if not intervals:
        intervals = [(0.0, len(wav) / sr)]

    windows = pack_windows(intervals, max_len=max_len, target_len=target_len)
    return slice_audio(wav, sr, windows)
