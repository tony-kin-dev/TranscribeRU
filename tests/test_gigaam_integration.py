"""Медленный интеграционный тест GigaAM: реально скачивает модель.

Помечен `@slow`, по умолчанию пропускается (`addopts = -m 'not slow'`).
Запуск: `pytest -m slow`. Требует установленных torch/transformers/silero-vad
и системного ffmpeg.
"""

import shutil
import subprocess

import numpy as np
import pytest

from transcribe_ru.audio import load
from transcribe_ru.core import transcribe
from transcribe_ru.engines.gigaam import GigaAMEngine


@pytest.mark.slow
def test_transcribe_short_wav_end_to_end(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg не установлен")

    # 3 секунды синуса — модель отработает без ошибок (текст может быть любым)
    wav_path = tmp_path / "tone.wav"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-f", "lavfi",
         "-i", "sine=frequency=220:duration=3", str(wav_path)],
        check=True, capture_output=True,
    )

    wav, sr = load(str(wav_path))
    assert sr == 16000
    assert wav.dtype == np.float32

    engine = GigaAMEngine()  # вариант по умолчанию e2e_rnnt
    result = transcribe(str(wav_path), engine, device="cpu")

    assert result.language == "ru"
    assert isinstance(result.segments, list)
    for seg in result.segments:
        assert isinstance(seg.text, str)
        assert seg.end > seg.start
