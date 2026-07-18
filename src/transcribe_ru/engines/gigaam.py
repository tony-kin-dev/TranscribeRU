"""Движок GigaAM-v3 через официальный пакет `gigaam` (Salute).

Публичный API `gigaam` принимает ПУТЬ к файлу (`model.transcribe(path)`), а не
массив. Ядро же владеет декодом и нарезкой (silero-vad) и отдаёт движку numpy-окно
<25с. Поэтому движок пишет окно во временный 16 кГц mono wav и зовёт публичный
`transcribe` (вариант B — устойчив к версиям, не лезет во внутренности модели).

pyannote НЕ требуется: он живёт в extra `gigaam[longform]`, а longform-режим
(gated `pyannote/segmentation-3.0` + HF_TOKEN) мы не используем — нарезку длинных
файлов делает silero-vad в ядре.

Тяжёлый импорт `gigaam` отложен в `_default_model_loader`, поэтому импорт модуля
движка (и его регистрация) дёшев и не тянет torch.
"""

from __future__ import annotations

import os
import tempfile
import wave

import numpy as np

from transcribe_ru.engines.base import Engine, register

# Варианты модели. В именах пакета gigaam — с префиксом версии: "v3_<variant>".
VARIANTS = ("e2e_rnnt", "e2e_ctc", "rnnt", "ctc")
DEFAULT_VARIANT = "e2e_rnnt"


# На Windows libtorch/sentencepiece (C++) не открывают не-ASCII пути, а кэш
# по умолчанию лежит в ~/.cache/gigaam — у русскоязычных это путь с кириллицей.
# Поэтому на Windows держим кэш в фиксированной ASCII-папке.
WINDOWS_CACHE_DIR = r"C:\gigaam_cache"


def _resolve_cache_dir(environ=None, os_name=None):
    """Папка кэша модели: env GIGAAM_CACHE_DIR > ASCII-папка на Windows > None."""
    environ = environ if environ is not None else os.environ
    os_name = os_name or os.name
    value = environ.get("GIGAAM_CACHE_DIR")
    if value:
        return value
    if os_name == "nt":
        return WINDOWS_CACHE_DIR
    return None


def _default_model_loader(model_name: str, device: str):
    """Загрузить модель пакетом gigaam на нужное устройство.

    fp16-энкодер включаем только на ускорителях (cuda/mps); на CPU fp16
    бессмысленен и местами не поддержан, поэтому там fp32. На Windows кэш
    модели направляем в ASCII-папку (см. `_resolve_cache_dir`).
    """
    import gigaam

    fp16_encoder = device not in ("cpu", None)
    kwargs = {"fp16_encoder": fp16_encoder, "device": device}
    cache_dir = _resolve_cache_dir()
    if cache_dir:
        kwargs["download_root"] = cache_dir
    return gigaam.load_model(model_name, **kwargs)


def _write_temp_wav(wav, sr: int) -> str:
    """Записать float32-окно во временный 16-бит mono wav; вернуть путь."""
    pcm = np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


@register("gigaam")
class GigaAMEngine(Engine):
    name = "gigaam"
    language = "ru"

    def __init__(self, variant: str = DEFAULT_VARIANT, *, model_loader=None):
        if variant not in VARIANTS:
            raise ValueError(
                f"Неизвестный вариант {variant!r}; допустимы: {', '.join(VARIANTS)}"
            )
        self.variant = variant
        self.model_name = f"v3_{variant}"
        self._model_loader = model_loader or _default_model_loader
        self._model = None

    def load(self, device: str) -> None:
        self._model = self._model_loader(self.model_name, device)

    def _run(self, wav, sr: int, **kwargs):
        """Записать окно во временный wav и позвать model.transcribe(path, **kwargs).

        Общий «танец» с temp-файлом для transcribe_segment и transcribe_words.
        """
        if self._model is None:
            raise RuntimeError("Движок не загружен; сначала вызовите load(device).")
        path = _write_temp_wav(wav, sr)
        try:
            return self._model.transcribe(path, **kwargs)
        finally:
            os.unlink(path)

    def transcribe_segment(self, wav, sr: int) -> str:
        return str(self._run(wav, sr)).strip()

    def transcribe_words(self, wav, sr: int):
        # word_timestamps=True зовём только на fine-пути; окна <25с → без longform/pyannote.
        return getattr(self._run(wav, sr, word_timestamps=True), "words", None)
