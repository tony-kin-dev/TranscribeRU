"""Тесты движка GigaAM. Загрузка модели инъектируется — torch не нужен."""

import pytest

from transcribe_ru.engines.base import get_engine
from transcribe_ru.engines.gigaam import VARIANTS, GigaAMEngine


class FakeModel:
    def __init__(self):
        self.last = None

    def transcribe(self, wav):
        self.last = wav
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
        eng.transcribe_segment([0.0, 0.1], 16000)


def test_load_then_transcribe_returns_stripped_text():
    captured = {}

    def loader(variant, device):
        captured["variant"] = variant
        captured["device"] = device
        return FakeModel()

    eng = GigaAMEngine(variant="e2e_ctc", model_loader=loader)
    eng.load("cpu")

    assert captured == {"variant": "e2e_ctc", "device": "cpu"}
    assert eng.transcribe_segment([0.1, 0.2, 0.3], 16000) == "распознанный текст"


def test_registered_in_engine_registry():
    # импорт модуля движка регистрирует его; torch не тянется (loader ленив)
    assert get_engine("gigaam") is GigaAMEngine


def test_all_variants_accepted():
    for v in VARIANTS:
        assert GigaAMEngine(variant=v).variant == v
