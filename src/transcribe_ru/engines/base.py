"""Интерфейс движков и реестр — точка расширения платформы.

Новый движок = подкласс `Engine` в отдельном файле + регистрация через
`@register("имя")`. Ядро работает только через этот интерфейс и не знает
о конкретных реализациях.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod

ENGINES: dict[str, type["Engine"]] = {}

# Встроенные движки подгружаются лениво по запросу (чтобы не тянуть torch
# и тяжёлые зависимости, пока движок реально не нужен).
_BUILTINS = {"gigaam": "transcribe_ru.engines.gigaam"}


class Engine(ABC):
    """Движок ASR: загрузка модели и распознавание одного окна <25с."""

    name: str

    @abstractmethod
    def load(self, device: str) -> None:
        """Ленивая загрузка модели на устройство."""

    @abstractmethod
    def transcribe_segment(self, wav, sr: int) -> str:
        """Распознать один кусок звука (<25с) → текст."""

    def transcribe_words(self, wav, sr: int):
        """Слова с временами ОТНОСИТЕЛЬНО начала окна (объекты с .text/.start/.end),
        либо None, если движок не умеет пословные таймстампы.

        Не абстрактный: движок обязателен только для `transcribe_segment`.
        None даёт ядру мягкий откат на крупную нарезку (granularity=coarse)."""
        return None


def register(name: str):
    """Декоратор: зарегистрировать класс движка под именем `name`."""

    def deco(cls: type[Engine]) -> type[Engine]:
        ENGINES[name] = cls
        return cls

    return deco


def get_engine(name: str) -> type[Engine]:
    """Вернуть класс движка по имени, лениво подгрузив встроенный модуль."""
    if name not in ENGINES and name in _BUILTINS:
        importlib.import_module(_BUILTINS[name])
    if name not in ENGINES:
        available = sorted(set(ENGINES) | set(_BUILTINS))
        raise ValueError(
            f"Неизвестный движок {name!r}; доступны: {', '.join(available)}"
        )
    return ENGINES[name]
