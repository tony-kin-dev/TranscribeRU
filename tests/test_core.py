"""Тесты ядра: Segment, TranscriptResult, оркестрация transcribe с fake-движком."""

import dataclasses

import pytest

from transcribe_ru.core import Segment, TranscriptResult, transcribe


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
