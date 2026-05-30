"""Тесты авто-выбора устройства. torch мокается фейковым модулем."""

import types

import pytest

from transcribe_ru.device import select_device


def _fake_torch(*, cuda: bool, mps: bool):
    """Фейковый модуль torch с заданной доступностью cuda/mps."""
    mod = types.SimpleNamespace()
    mod.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    mod.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: mps)
    )
    return mod


def test_auto_prefers_cuda_when_available():
    torch = _fake_torch(cuda=True, mps=True)
    assert select_device("auto", torch_module=torch) == "cuda"


def test_auto_falls_back_to_mps_when_no_cuda():
    torch = _fake_torch(cuda=False, mps=True)
    assert select_device("auto", torch_module=torch) == "mps"


def test_auto_falls_back_to_cpu_when_nothing():
    torch = _fake_torch(cuda=False, mps=False)
    assert select_device("auto", torch_module=torch) == "cpu"


def test_explicit_device_returned_as_is():
    torch = _fake_torch(cuda=False, mps=False)
    assert select_device("cuda", torch_module=torch) == "cuda"


def test_unknown_device_raises():
    torch = _fake_torch(cuda=False, mps=False)
    with pytest.raises(ValueError):
        select_device("gpu", torch_module=torch)
