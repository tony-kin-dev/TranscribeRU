"""Тесты открытия папки с результатом — команда зависит от ОС."""

from transcribe_ru import gui


def _capture():
    calls = []
    return calls, (lambda cmd: calls.append(cmd))


def test_macos_uses_open_dash_r():
    calls, run = _capture()
    gui.reveal_in_file_manager("/a/b.txt", platform_name="darwin", run=run)
    assert calls == [["open", "-R", "/a/b.txt"]]


def test_windows_uses_explorer_select():
    calls, run = _capture()
    gui.reveal_in_file_manager(r"C:\a\b.txt", platform_name="win32", run=run)
    assert calls == [["explorer", r"/select,C:\a\b.txt"]]


def test_linux_opens_parent_directory():
    calls, run = _capture()
    gui.reveal_in_file_manager("/a/b.txt", platform_name="linux", run=run)
    assert calls == [["xdg-open", "/a"]]


def test_default_platform_is_current_system():
    # без явной платформы берётся sys.platform — команда всё равно формируется
    calls, run = _capture()
    gui.reveal_in_file_manager("/a/b.txt", run=run)
    assert len(calls) == 1
    assert calls[0][0] in ("open", "explorer", "xdg-open")
