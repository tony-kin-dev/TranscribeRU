"""Тесты ядра: Segment, TranscriptResult, оркестрация transcribe с fake-движком."""

import dataclasses

import pytest

from transcribe_ru.core import Segment, TranscriptResult, group_words, transcribe


def test_segment_is_frozen_dataclass():
    seg = Segment(0.0, 1.0, "привет")
    assert (seg.start, seg.end, seg.text) == (0.0, 1.0, "привет")
    with pytest.raises(dataclasses.FrozenInstanceError):
        seg.text = "другое"


def test_segment_equality():
    assert Segment(0.0, 1.0, "a") == Segment(0.0, 1.0, "a")


class FakeEngine:
    """Движок для тестов: помнит device, на каждый кусок отдаёт его индекс."""

    name = "fake"
    language = "ru"

    def __init__(self):
        self.loaded_device = None
        self.calls = []

    def load(self, device):
        self.loaded_device = device

    def transcribe_segment(self, wav, sr):
        self.calls.append((wav, sr))
        return f"кусок {len(self.calls)}"


def _fake_chunks():
    # (wav_chunk, (start, end)) — wav как простой плейсхолдер
    return [
        (["w1"], (0.0, 5.0)),
        (["w2"], (5.0, 11.5)),
    ]


def test_transcribe_orchestrates_load_segment_engine():
    engine = FakeEngine()
    progress = []

    result = transcribe(
        "dummy.opus",
        engine,
        device="cpu",
        on_progress=lambda done, total: progress.append((done, total)),
        load_audio=lambda path: (["full"], 16000),
        segment_fn=lambda wav, sr: _fake_chunks(),
    )

    assert isinstance(result, TranscriptResult)
    assert engine.loaded_device == "cpu"
    assert result.language == "ru"
    assert result.segments == [
        Segment(0.0, 5.0, "кусок 1"),
        Segment(5.0, 11.5, "кусок 2"),
    ]
    # движок звался с переданным sr
    assert engine.calls == [(["w1"], 16000), (["w2"], 16000)]
    # прогресс доходит до полного
    assert progress[-1] == (2, 2)


def test_transcribe_skips_empty_text_segments():
    engine = FakeEngine()
    engine.transcribe_segment = lambda wav, sr: "   "  # только пробелы

    result = transcribe(
        "dummy.opus",
        engine,
        device="cpu",
        load_audio=lambda path: (["full"], 16000),
        segment_fn=lambda wav, sr: _fake_chunks(),
    )
    assert result.segments == []


def test_transcribe_works_without_progress_callback():
    engine = FakeEngine()
    result = transcribe(
        "dummy.opus",
        engine,
        device="cpu",
        load_audio=lambda path: (["full"], 16000),
        segment_fn=lambda wav, sr: _fake_chunks(),
    )
    assert len(result.segments) == 2


class _W:
    """Слово с временами относительно окна — плейсхолдер для group_words."""

    def __init__(self, text, start, end):
        self.text, self.start, self.end = text, start, end


class FineEngine(FakeEngine):
    """Движок с пословными таймстампами для проверки режима fine."""

    def transcribe_words(self, wav, sr):
        return [
            _W("Привет,", 0.0, 0.5), _W("мир.", 0.6, 1.6),
            _W("Как", 1.7, 2.0), _W("дела?", 2.1, 2.8),
        ]


def test_fine_splits_words_into_cues_with_offset():
    engine = FineEngine()
    result = transcribe(
        "dummy.opus",
        engine,
        device="cpu",
        granularity="fine",
        load_audio=lambda path: (["full"], 16000),
        # одно окно со стартом 10.0 — проверяем прибавление offset
        segment_fn=lambda wav, sr: [(["w"], (10.0, 30.0))],
    )
    assert [(round(s.start, 1), round(s.end, 1), s.text) for s in result.segments] == [
        (10.0, 11.6, "Привет, мир."),   # разрыв по концу предложения + offset
        (11.7, 12.8, "Как дела?"),
    ]


def test_coarse_ignores_word_timestamps():
    # granularity=coarse (дефолт) не должен дёргать transcribe_words
    engine = FineEngine()
    engine.transcribe_words = lambda wav, sr: (_ for _ in ()).throw(
        AssertionError("transcribe_words не должен вызываться в coarse")
    )
    result = transcribe(
        "dummy.opus",
        engine,
        device="cpu",
        load_audio=lambda path: (["full"], 16000),
        segment_fn=lambda wav, sr: _fake_chunks(),
    )
    assert len(result.segments) == 2  # обычная покусочная нарезка


def test_group_words_degrades_without_punctuation():
    # без пунктуации режем по паузе (2.5с > gap); слова не крошечные → не склеиваются
    cues = group_words([_W("привет", 0.0, 1.5), _W("здравствуй", 4.0, 5.5)], gap=0.6)
    assert [(round(c.start, 1), round(c.end, 1), c.text) for c in cues] == [
        (0.0, 1.5, "привет"),
        (4.0, 5.5, "здравствуй"),
    ]


def test_group_words_merges_short_interjections():
    # «Угу.» между репликами не должно стать отдельным мелькающим тайм-кодом —
    # приклеивается к соседней реплике
    words = [
        _W("Я", 0.0, 0.3), _W("предлагаю.", 0.4, 1.6),        # реплика 1
        _W("Угу.", 2.0, 2.6),                                 # короткая вставка
        _W("Хорошо,", 4.0, 4.6), _W("давайте.", 4.7, 6.0),    # реплика 2
    ]
    cues = group_words(words, gap=0.6)
    texts = [c.text for c in cues]
    assert "Угу." not in texts               # отдельной реплики «Угу.» нет
    assert any("Угу." in t for t in texts)   # но текст сохранён внутри соседней
    assert len(cues) == 2


def test_group_words_merges_leading_short_into_next():
    # короткая вставка в самом начале (предыдущей нет) → приклеивается к следующей
    cues = group_words([_W("Да.", 0.0, 0.4), _W("Поехали", 2.0, 4.0)], gap=0.6)
    assert len(cues) == 1
    assert cues[0].text == "Да. Поехали"
    assert (round(cues[0].start, 1), round(cues[0].end, 1)) == (0.0, 4.0)
