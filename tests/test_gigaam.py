"""Тесты движка GigaAM. Загрузка модели инъектируется — пакет gigaam не нужен.

Подача аудио (вариант B): окно пишется во временный wav, движок зовёт публичный
`model.transcribe(path)`. FakeModel реально читает записанный wav, проверяя, что
движок отдаёт корректный 16 кГц mono int16 файл.
"""

import os
import wave

import numpy as np
import pytest

from transcribe_ru.engines.base import get_engine
from transcribe_ru.engines.gigaam import VARIANTS, GigaAMEngine


class FakeModel:
    def __init__(self):
        self.path = None
        self.sr = None
        self.channels = None
        self.sampwidth = None
        self.path_existed = None

    def transcribe(self, path):
        # путь должен существовать в момент вызова и быть валидным wav
        self.path_existed = os.path.exists(path)
        with wave.open(path, "rb") as w:
            self.sr = w.getframerate()
            self.channels = w.getnchannels()
            self.sampwidth = w.getsampwidth()
        self.path = path
        return "  распознанный текст  "


def test_default_variant_is_e2e_rnnt():
    assert GigaAMEngine().variant == "e2e_rnnt"


def test_invalid_variant_raises():
    with pytest.raises(ValueError):
        GigaAMEngine(variant="banana")


def test_name_and_language():
    eng = GigaAMEngine()
    assert eng.name == "gigaam"
    assert eng.language == "ru"


def test_transcribe_before_load_raises():
    eng = GigaAMEngine()
    with pytest.raises(RuntimeError, match="load"):
        eng.transcribe_segment(np.zeros(16000, dtype=np.float32), 16000)


def test_load_maps_variant_to_v3_model_name():
    captured = {}

    def loader(model_name, device):
        captured["model_name"] = model_name
        captured["device"] = device
        return FakeModel()

    GigaAMEngine(variant="e2e_ctc", model_loader=loader).load("cpu")
    assert captured == {"model_name": "v3_e2e_ctc", "device": "cpu"}


def test_transcribe_writes_valid_wav_and_returns_stripped_text():
    model = FakeModel()
    eng = GigaAMEngine(model_loader=lambda mn, dev: model)
    eng.load("cpu")

    wav = np.zeros(16000, dtype=np.float32)
    text = eng.transcribe_segment(wav, 16000)

    assert text == "распознанный текст"
    assert model.path_existed is True
    assert model.sr == 16000
    assert model.channels == 1
    assert model.sampwidth == 2  # int16


def test_transcribe_cleans_up_temp_file():
    model = FakeModel()
    eng = GigaAMEngine(model_loader=lambda mn, dev: model)
    eng.load("cpu")
    eng.transcribe_segment(np.zeros(8000, dtype=np.float32), 16000)
    # временный файл удалён после распознавания
    assert not os.path.exists(model.path)


def test_registered_in_engine_registry():
    assert get_engine("gigaam") is GigaAMEngine


def test_all_variants_accepted():
    for v in VARIANTS:
        assert GigaAMEngine(variant=v).variant == v
