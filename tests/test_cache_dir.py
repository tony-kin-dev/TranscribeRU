"""Тесты выбора папки кэша модели (Windows + кириллица → ASCII-папка)."""

from transcribe_ru.engines.gigaam import WINDOWS_CACHE_DIR, _resolve_cache_dir


def test_env_var_wins_everywhere():
    assert _resolve_cache_dir({"GIGAAM_CACHE_DIR": r"D:\c"}, "nt") == r"D:\c"
    assert _resolve_cache_dir({"GIGAAM_CACHE_DIR": "/data/c"}, "posix") == "/data/c"


def test_windows_default_is_ascii_folder():
    # без env на Windows — фиксированная ASCII-папка (libtorch/sentencepiece
    # не открывают кириллические пути)
    assert _resolve_cache_dir({}, "nt") == WINDOWS_CACHE_DIR
    assert WINDOWS_CACHE_DIR.isascii()


def test_posix_default_is_none():
    # на macOS/Linux отдаём выбор пакету gigaam (~/.cache/gigaam)
    assert _resolve_cache_dir({}, "posix") is None


def test_empty_env_value_ignored():
    assert _resolve_cache_dir({"GIGAAM_CACHE_DIR": ""}, "nt") == WINDOWS_CACHE_DIR
