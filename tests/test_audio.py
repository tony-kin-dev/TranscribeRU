"""Тесты загрузки аудио через ffmpeg. Входной файл синтезируется ffmpeg'ом."""

import shutil
import subprocess

import numpy as np
import pytest

from transcribe_ru.audio import load

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg не установлен"
)


@pytest.fixture
def sine_wav(tmp_path):
    """2 секунды синуса 440 Гц в WAV-файле (через ffmpeg)."""
    path = tmp_path / "tone.wav"
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-y", "-f", "lavfi",
            "-i", "sine=frequency=440:duration=2", str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def test_load_returns_float32_mono_16k(sine_wav):
    wav, sr = load(str(sine_wav))
    assert sr == 16000
    assert wav.dtype == np.float32
    assert wav.ndim == 1
    # ~2 секунды при 16 кГц
    assert abs(len(wav) - 2 * 16000) < 16000 * 0.1


def test_load_values_in_unit_range(sine_wav):
    wav, _ = load(str(sine_wav))
    assert wav.max() <= 1.0
    assert wav.min() >= -1.0
    assert wav.max() > 0.1  # синус не пустой


def test_missing_ffmpeg_raises_helpful_error(sine_wav):
    with pytest.raises(RuntimeError, match="ffmpeg"):
        load(str(sine_wav), ffmpeg_bin="ffmpeg-does-not-exist-xyz")


def test_bad_input_raises(tmp_path):
    bad = tmp_path / "not-audio.wav"
    bad.write_text("это не аудио")
    with pytest.raises(RuntimeError):
        load(str(bad))
