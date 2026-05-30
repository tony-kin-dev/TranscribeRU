"""Тесты интерфейса движков и реестра. Без torch — встроенные не импортируются."""

import pytest

from transcribe_ru.engines import base
from transcribe_ru.engines.base import Engine, get_engine, register


def test_engine_cannot_be_instantiated_without_methods():
    class Incomplete(Engine):
        name = "incomplete"

    with pytest.raises(TypeError):
        Incomplete()


def test_concrete_engine_can_be_instantiated():
    class Ok(Engine):
        name = "ok"

        def load(self, device):
            pass

        def transcribe_segment(self, wav, sr):
            return ""

    Ok()  # не должно бросить


def test_register_adds_engine_and_get_engine_returns_it():
    @register("fake_test_engine")
    class FakeEngine(Engine):
        name = "fake_test_engine"

        def load(self, device):
            pass

        def transcribe_segment(self, wav, sr):
            return ""

    try:
        assert get_engine("fake_test_engine") is FakeEngine
    finally:
        base.ENGINES.pop("fake_test_engine", None)


def test_get_unknown_engine_raises_value_error_without_importing_builtins():
    # неизвестное имя не должно триггерить импорт gigaam (тянет torch)
    with pytest.raises(ValueError, match="Неизвестный движок"):
        get_engine("no-such-engine")
