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


def group_words(
    words, *, offset: float = 0.0, max_dur: float = 7.0, gap: float = 0.6,
    max_chars: int = 84,
) -> list[Segment]:
    """Слова (с временами относительно окна) → короткие реплики-`Segment`.

    Жадно копит слова в реплику и разрывает её при: конце предложения
    (последний символ слова в `.!?…`), паузе > `gap`, длительности > `max_dur`
    или длине > `max_chars`. `offset` (начало окна) прибавляется к временам.
    Первое слово реплики не режем — одиночное длинное слово выходит целиком.

    Без пунктуации (варианты rnnt/ctc) разрыв по предложению не срабатывает —
    остаются пределы по паузе/длительности/длине (корректная деградация).
    Пороги — калибровочные ручки, в CLI/GUI не выносятся (# ponytail: до запроса).
    """
    segments: list[Segment] = []
    buf: list = []

    def flush():
        if not buf:
            return
        text = " ".join(w.text for w in buf).strip()
        if text:
            segments.append(Segment(buf[0].start + offset, buf[-1].end + offset, text))
        buf.clear()

    for w in words:
        if buf:
            cur = " ".join(x.text for x in buf)
            over_dur = (w.end - buf[0].start) > max_dur
            over_chars = len(cur) + 1 + len(w.text) > max_chars
            paused = (w.start - buf[-1].end) > gap
            if over_dur or over_chars or paused:
                flush()
        buf.append(w)
        if w.text and w.text[-1] in ".!?…":
            flush()
    flush()
    return segments


def transcribe(
    audio_path,
    engine,
    *,
    device: str = "auto",
    granularity: str = "coarse",
    on_progress=None,
    load_audio=None,
    segment_fn=None,
) -> TranscriptResult:
    """Транскрибировать аудиофайл через переданный движок.

    Шаги: загрузка аудио (16 kHz mono float32) → нарезка на окна <25с →
    распознавание каждого окна движком → сборка `Segment`.

    `granularity`: `coarse` — один сегмент на окно (по умолчанию); `fine` —
    короткие реплики ~5-7с через пословные таймстампы движка (для субтитров).

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
        words = engine.transcribe_words(chunk, sr) if granularity == "fine" else None
        if words:
            segments.extend(group_words(words, offset=start))
        else:
            # coarse — или fine на пустом/тихом окне (мягкий откат;
            # # ponytail: второй прогон модели только на окнах без слов)
            text = engine.transcribe_segment(chunk, sr).strip()
            if text:
                segments.append(Segment(start, end, text))
        if on_progress is not None:
            on_progress(done, total)

    language = getattr(engine, "language", None)
    return TranscriptResult(segments=segments, language=language)
