"""Тесты соответствия русских подписей и технических значений в GUI.

Импорт gui не создаёт окно (tkinter импортируется лениво внутри методов),
поэтому маппинг подписей проверяется без дисплея.
"""

from transcribe_ru import gui
from transcribe_ru.cli import FORMATS
from transcribe_ru.engines.gigaam import VARIANTS


def _values(choices):
    return [value for _, value in choices]


def _labels(choices):
    return [label for label, _ in choices]


def test_format_choices_cover_all_formats():
    assert set(_values(gui.FORMAT_CHOICES)) == set(FORMATS)


def test_variant_choices_cover_all_variants():
    assert set(_values(gui.VARIANT_CHOICES)) == set(VARIANTS)


def test_device_choices_cover_all_devices():
    assert set(_values(gui.DEVICE_CHOICES)) == {"auto", "cuda", "mps", "cpu"}


def test_defaults_are_first_choice():
    # первая подпись — значение по умолчанию
    assert gui.label_to_value(gui.FORMAT_CHOICES, _labels(gui.FORMAT_CHOICES)[0]) == "txt_timecoded"
    assert gui.label_to_value(gui.VARIANT_CHOICES, _labels(gui.VARIANT_CHOICES)[0]) == "e2e_rnnt"
    assert gui.label_to_value(gui.DEVICE_CHOICES, _labels(gui.DEVICE_CHOICES)[0]) == "auto"


def test_labels_are_russian_not_raw_value():
    # подпись не должна совпадать с техническим значением (т.е. она расшифрована)
    for choices in (gui.FORMAT_CHOICES, gui.VARIANT_CHOICES, gui.DEVICE_CHOICES):
        for label, value in choices:
            assert label != value
            assert any("а" <= ch <= "я" or "А" <= ch <= "Я" for ch in label)


def test_label_to_value_roundtrip():
    for label, value in gui.VARIANT_CHOICES:
        assert gui.label_to_value(gui.VARIANT_CHOICES, label) == value


def test_label_to_value_unknown_raises():
    import pytest

    with pytest.raises(KeyError):
        gui.label_to_value(gui.FORMAT_CHOICES, "нет такой подписи")


def test_media_filetypes_include_audio_and_video():
    patterns = " ".join(pattern for _, pattern in gui.MEDIA_TYPES).lower()
    # аудио
    for ext in ("opus", "mp3", "wav", "m4a"):
        assert ext in patterns
    # видео
    for ext in ("mp4", "mov", "mkv", "avi", "webm"):
        assert ext in patterns


def test_media_filetypes_have_all_files_fallback():
    assert any(pattern == "*" for _, pattern in gui.MEDIA_TYPES)
