"""Графическое окно (tkinter) поверх ядра.

Тонкий слой: собирает параметры из виджетов и вызывает `runner.transcribe_to_file`
в фоновом потоке. tkinter не потокобезопасен, поэтому фоновый поток только кладёт
события в очередь, а главный поток рисует их через `root.after`.

Запуск: `transcribe-ru-gui` (или двойным кликом по ярлыку из `--install-shortcut`).
Логики транскрипции здесь нет — она в `runner`/`core`, покрыта тестами.
"""

from __future__ import annotations

import argparse
import os
import queue
import stat
import sys
import threading
from pathlib import Path

# Пары (русская подпись для пользователя, техническое значение для движка).
# Первый элемент в каждом списке — значение по умолчанию.
FORMAT_CHOICES = [
    ("Текст с тайм-кодами (ЧЧ:ММ:СС)", "txt_timecoded"),
    ("Сплошной текст без тайм-кодов", "txt_plain"),
    ("Субтитры (.srt)", "srt"),
    ("JSON (для разработчиков)", "json"),
]
VARIANT_CHOICES = [
    ("С пунктуацией, точнее (рекомендуется)", "e2e_rnnt"),
    ("С пунктуацией, быстрее", "e2e_ctc"),
    ("Без пунктуации, точнее", "rnnt"),
    ("Без пунктуации, быстрее", "ctc"),
]
DEVICE_CHOICES = [
    ("Автоматически (рекомендуется)", "auto"),
    ("Видеокарта NVIDIA (CUDA)", "cuda"),
    ("Чип Apple (MPS)", "mps"),
    ("Процессор (CPU, медленно)", "cpu"),
]

# ffmpeg извлекает аудиодорожку и из видео, поэтому принимаем и медиа-видео.
_AUDIO_PATTERN = "*.opus *.m4a *.mp3 *.wav *.aiff *.flac *.ogg *.aac *.wma"
_VIDEO_PATTERN = "*.mp4 *.mov *.mkv *.avi *.webm *.m4v *.flv *.wmv *.mpeg *.mpg"
MEDIA_TYPES = [
    ("Медиа (аудио и видео)", f"{_AUDIO_PATTERN} {_VIDEO_PATTERN}"),
    ("Аудио", _AUDIO_PATTERN),
    ("Видео", _VIDEO_PATTERN),
    ("Все файлы", "*"),
]


def label_to_value(choices, label: str) -> str:
    """Вернуть техническое значение по выбранной русской подписи."""
    for lab, value in choices:
        if lab == label:
            return value
    raise KeyError(label)


def reveal_in_file_manager(path, *, platform_name=None, run=None) -> None:
    """Открыть папку с файлом в проводнике и по возможности выделить файл.

    Команда зависит от ОС: macOS — `open -R`, Windows — `explorer /select,`,
    прочее — `xdg-open` на родительскую папку. `platform_name`/`run` инъектируются
    в тестах; по умолчанию берутся `sys.platform` и `subprocess.run`.
    """
    platform_name = platform_name or sys.platform
    if run is None:
        import subprocess

        def run(cmd):
            subprocess.run(cmd, check=False)

    if platform_name == "darwin":
        run(["open", "-R", path])
    elif platform_name.startswith("win"):
        run(["explorer", f"/select,{path}"])
    else:
        run(["xdg-open", os.path.dirname(path) or "."])


class TranscribeApp:
    """Окно: выбор файла, настройки, запуск и прогресс."""

    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.events: "queue.Queue" = queue.Queue()
        self.worker: threading.Thread | None = None

        root.title("TranscribeRU")
        root.resizable(False, False)

        # в переменных хранятся русские ПОДПИСИ; значение для движка получаем
        # через label_to_value при запуске
        self.audio_path = tk.StringVar()
        self.fmt = tk.StringVar(value=FORMAT_CHOICES[0][0])
        self.variant = tk.StringVar(value=VARIANT_CHOICES[0][0])
        self.device = tk.StringVar(value=DEVICE_CHOICES[0][0])
        self.status = tk.StringVar(value="Выберите аудио- или видеофайл")

        self._build()

    def _build(self):
        tk, ttk = self.tk, self.ttk
        frm = ttk.Frame(self.root, padding=12)
        frm.grid(sticky="nsew")

        ttk.Label(frm, text="Файл:").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.audio_path, width=36, state="readonly").grid(
            row=0, column=1, sticky="we", padx=4
        )
        ttk.Button(frm, text="Выбрать…", command=self._choose_file).grid(row=0, column=2)

        self._combo(frm, "Формат:", self.fmt, [c[0] for c in FORMAT_CHOICES], 1)
        self._combo(frm, "Вариант:", self.variant, [c[0] for c in VARIANT_CHOICES], 2)
        self._combo(frm, "Устройство:", self.device, [c[0] for c in DEVICE_CHOICES], 3)

        self.run_btn = ttk.Button(frm, text="Транскрибировать", command=self._start)
        self.run_btn.grid(row=4, column=0, columnspan=3, pady=(10, 4), sticky="we")

        self.bar = ttk.Progressbar(frm, mode="determinate", maximum=1)
        self.bar.grid(row=5, column=0, columnspan=3, sticky="we")
        ttk.Label(frm, textvariable=self.status, wraplength=320).grid(
            row=6, column=0, columnspan=3, sticky="w", pady=(6, 0)
        )

    def _combo(self, frm, label, var, values, row):
        ttk = self.ttk
        ttk.Label(frm, text=label).grid(row=row, column=0, sticky="w", pady=2)
        box = ttk.Combobox(frm, textvariable=var, values=list(values), state="readonly")
        box.grid(row=row, column=1, columnspan=2, sticky="we", padx=4, pady=2)

    def _choose_file(self):
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            title="Выберите аудио- или видеофайл", filetypes=MEDIA_TYPES
        )
        if path:
            self.audio_path.set(path)
            self.status.set("Готов к запуску")

    def _start(self):
        from tkinter import messagebox

        if self.worker and self.worker.is_alive():
            return
        audio = self.audio_path.get().strip()
        if not audio:
            messagebox.showwarning("Нет файла", "Сначала выберите аудио- или видеофайл.")
            return

        self.run_btn.state(["disabled"])
        self.bar.configure(mode="determinate", maximum=1, value=0)
        self.status.set("Подготовка модели и аудио…")

        params = dict(
            audio=audio,
            fmt=label_to_value(FORMAT_CHOICES, self.fmt.get()),
            variant=label_to_value(VARIANT_CHOICES, self.variant.get()),
            device=label_to_value(DEVICE_CHOICES, self.device.get()),
        )
        self.worker = threading.Thread(target=self._run_job, args=(params,), daemon=True)
        self.worker.start()
        self.root.after(100, self._poll)

    def _run_job(self, params):
        """Фоновый поток: гоняет транскрипцию, шлёт события в очередь."""
        from transcribe_ru.runner import transcribe_to_file

        try:
            out = transcribe_to_file(
                params["audio"],
                variant=params["variant"],
                device=params["device"],
                fmt=params["fmt"],
                on_progress=lambda d, t: self.events.put(("progress", d, t)),
            )
            self.events.put(("done", str(out)))
        except Exception as exc:  # noqa: BLE001 — показываем пользователю любую ошибку
            self.events.put(("error", str(exc)))

    def _poll(self):
        """Главный поток: рисует события из очереди."""
        from tkinter import messagebox

        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]
                if kind == "progress":
                    _, done, total = event
                    self.bar.configure(maximum=max(total, 1), value=done)
                    self.status.set(f"Распознавание: {done}/{total}")
                elif kind == "done":
                    self.bar.configure(value=self.bar["maximum"])
                    self.status.set(f"Готово: {event[1]}")
                    self.run_btn.state(["!disabled"])
                    # открыть папку с результатом (не критично, если не вышло)
                    try:
                        reveal_in_file_manager(event[1])
                    except Exception:  # noqa: BLE001
                        pass
                    return
                elif kind == "error":
                    self.status.set("Ошибка")
                    self.run_btn.state(["!disabled"])
                    messagebox.showerror("Ошибка транскрипции", event[1])
                    return
        except queue.Empty:
            pass
        self.root.after(100, self._poll)


def run_gui() -> int:
    import tkinter as tk

    root = tk.Tk()
    TranscribeApp(root)
    root.mainloop()
    return 0


def install_shortcut() -> Path:
    """Создать на Рабочем столе ярлык `.command` для запуска двойным кликом."""
    desktop = Path.home() / "Desktop"
    desktop.mkdir(parents=True, exist_ok=True)
    shortcut = desktop / "TranscribeRU.command"
    shortcut.write_text(
        f'#!/bin/bash\nexec "{sys.executable}" -m transcribe_ru.gui\n',
        encoding="utf-8",
    )
    shortcut.chmod(shortcut.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return shortcut


def _write_crash_log(text: str, log_dir=None) -> Path:
    """Записать текст ошибки в error.log; вернуть путь к файлу."""
    if log_dir is None:
        base = os.getenv("LOCALAPPDATA") or str(Path.home())
        log_dir = Path(base) / "TranscribeRU"
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "error.log"
    log_file.write_text(text, encoding="utf-8")
    return log_file


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="transcribe-ru-gui", description="Окно для запуска транскрипции."
    )
    parser.add_argument(
        "--install-shortcut",
        action="store_true",
        help="Создать ярлык TranscribeRU.command на Рабочем столе и выйти.",
    )
    args = parser.parse_args(argv)

    from transcribe_ru.certs import configure_ssl

    configure_ssl()  # лечит CERTIFICATE_VERIFY_FAILED при скачивании модели

    if args.install_shortcut:
        path = install_shortcut()
        print(f"Ярлык создан: {path}")
        print("Двойной клик по нему открывает окно (один раз подтвердите запуск в macOS).")
        return 0

    # Ярлык на Windows запускается через pythonw — без консоли исключение
    # пропадёт молча. Пишем трейсбек в лог и показываем окно с путём.
    try:
        return run_gui()
    except Exception:
        import traceback

        log = _write_crash_log(traceback.format_exc())
        try:
            from tkinter import messagebox

            messagebox.showerror(
                "TranscribeRU", f"Ошибка запуска. Подробности в файле:\n{log}"
            )
        except Exception:
            pass
        raise


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
