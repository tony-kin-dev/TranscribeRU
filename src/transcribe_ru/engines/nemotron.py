"""Движок NVIDIA Nemotron 3.5 ASR Streaming (0.6B) через `transformers` (>=5.13).

Мультиязычная RNN-T модель с пунктуацией и заглавными «из коробки». Путь через
transformers, а не NeMo: NeMo официально только Linux+NVIDIA, а transformers
работает и на Apple MPS, и на CPU. Веса (~2.5 ГБ) качаются в кэш Hugging Face
при первом запуске.

Варианты — язык распознавания: `auto` (модель сама определяет язык, удобно для
речи с английскими вставками) или `ru-RU` (жёстко русский; по FLEURS NVIDIA чуть
точнее на чистом русском). Пословных таймстампов нет → fine-режим ядра мягко
откатывается на крупную нарезку.

Тяжёлый импорт transformers отложен в `_default_model_loader`.
"""

from __future__ import annotations

import numpy as np

from transcribe_ru.engines.base import Engine, register

MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"
VARIANTS = ("auto", "ru-RU")
DEFAULT_VARIANT = "auto"


def _default_model_loader(device: str):
    from transformers import AutoModelForRNNT, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForRNNT.from_pretrained(MODEL_ID).to(device)
    return processor, model


@register("nemotron")
class NemotronEngine(Engine):
    name = "nemotron"

    def __init__(self, variant: str | None = None, *, model_loader=None):
        variant = variant or DEFAULT_VARIANT
        if variant not in VARIANTS:
            raise ValueError(
                f"Неизвестный вариант {variant!r}; допустимы: {', '.join(VARIANTS)}"
            )
        self.variant = variant
        # язык результата: известен только при жёстко заданном языке
        self.language = "ru" if variant == "ru-RU" else None
        self._model_loader = model_loader or _default_model_loader
        self._processor = self._model = None
        self._device = None

    def load(self, device: str) -> None:
        self._processor, self._model = self._model_loader(device)
        self._device = device

    def transcribe_segment(self, wav, sr: int) -> str:
        if self._model is None:
            raise RuntimeError("Движок не загружен; сначала вызовите load(device).")
        inputs = self._processor(
            np.asarray(wav, dtype=np.float32), sampling_rate=sr, language=self.variant
        ).to(self._device)
        out = self._model.generate(**inputs, return_dict_in_generate=True)
        decoded = self._processor.decode(out.sequences, skip_special_tokens=True)
        text = " ".join(decoded) if isinstance(decoded, list) else str(decoded)
        return " ".join(text.split())  # модель ставит двойные пробелы между фразами
