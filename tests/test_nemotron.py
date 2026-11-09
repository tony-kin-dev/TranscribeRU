"""Движок Nemotron: передаёт язык процессору и чистит пробелы (без скачивания модели)."""

import numpy as np
import pytest

from transcribe_ru.engines.base import get_engine
from transcribe_ru.engines.nemotron import NemotronEngine


class _Inputs(dict):
    def to(self, device):
        self["device"] = device
        return self


class _FakeProcessor:
    def __call__(self, wav, sampling_rate, language):
        self.language = language
        return _Inputs()

    def decode(self, sequences, skip_special_tokens):
        return [" Привет!  Как дела? "]


class _FakeModel:
    def generate(self, **kwargs):
        return type("Out", (), {"sequences": None})()


def test_nemotron_transcribes_with_language_and_normalizes_spaces():
    proc = _FakeProcessor()
    eng = NemotronEngine("ru-RU", model_loader=lambda device: (proc, _FakeModel()))
    eng.load("cpu")
    assert eng.transcribe_segment(np.zeros(16000), 16000) == "Привет! Как дела?"
    assert proc.language == "ru-RU"
    assert eng.language == "ru"


def test_nemotron_default_variant_and_registry():
    assert get_engine("nemotron") is NemotronEngine
    assert NemotronEngine().variant == "auto"
    with pytest.raises(ValueError):
        NemotronEngine("e2e_rnnt")
