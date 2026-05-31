"""Тесты записи краш-лога GUI (ярлык через pythonw гасит traceback)."""

from transcribe_ru.gui import _write_crash_log


def test_crash_log_written_to_dir(tmp_path):
    log = _write_crash_log("трейсбек тут", log_dir=tmp_path / "TranscribeRU")
    assert log.name == "error.log"
    assert log.exists()
    assert log.read_text(encoding="utf-8") == "трейсбек тут"


def test_crash_log_creates_missing_dirs(tmp_path):
    target = tmp_path / "a" / "b" / "TranscribeRU"
    log = _write_crash_log("x", log_dir=target)
    assert log.exists()
    assert log.parent == target
