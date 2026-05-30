"""Ядро платформы: модель результата и оркестрация нарезка → движок → сборка.

Ядро не знает о конкретном движке — общается через интерфейс `Engine`
(`load`, `transcribe_segment`). Нарезкой и сборкой сегментов владеет ядро.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    """Один распознанный фрагмент с границами в секундах."""

    start: float
    end: float
    text: str


@dataclass
class TranscriptResult:
    """Результат транскрипции: сегменты и (опционально) язык."""

    segments: list[Segment]
    language: str | None = None


def transcribe(
    audio_path,
    engine,
    *,
    device: str = "auto",
    on_progress=None,
    load_audio=None,
    segment_fn=None,
) -> TranscriptResult:
    """Транскрибировать аудиофайл через переданный движок.

    Шаги: загрузка аудио (16 kHz mono float32) → нарезка на окна <25с →
    распознавание каждого окна движком → сборка `Segment`.

    `load_audio`/`segment_fn` инъектируются в тестах; по умолчанию берутся
    из модулей `audio` и `segmentation`.
    """
    if load_audio is None:
        from transcribe_ru.audio import load as load_audio
    if segment_fn is None:
        from transcribe_ru.segmentation import segment as segment_fn

    wav, sr = load_audio(audio_path)
    chunks = segment_fn(wav, sr)

    engine.load(device)

    total = len(chunks)
    segments: list[Segment] = []
    for done, (chunk, (start, end)) in enumerate(chunks, start=1):
        text = engine.transcribe_segment(chunk, sr).strip()
        if text:
            segments.append(Segment(start, end, text))
        if on_progress is not None:
            on_progress(done, total)

    language = getattr(engine, "language", None)
    return TranscriptResult(segments=segments, language=language)
