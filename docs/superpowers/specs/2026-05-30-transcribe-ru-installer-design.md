# TranscribeRU — установщики в один шаг (macOS / Windows) — дизайн

Дата: 2026-05-30
Статус: утверждён к реализации

## 1. Назначение

Дать конечному пользователю один скрипт под его ОС, который по ссылке с GitHub
поставит **всё**: пакетный менеджер (если нужно), Python 3.12, ffmpeg, git, сам
проект с зависимостями в виртуальном окружении, и положит иконку запуска на
Рабочий стол. Принцип каждого шага: **что можно — ставим автоматически; чего
нельзя без участия владельца — печатаем понятное сообщение со ссылкой и выходим.**

Целевые ОС: macOS и Windows. Linux — вне рамок.

## 2. Артефакты

```
scripts/
├── install.command   # macOS: bash, запускается двойным кликом или curl|bash
├── install.ps1       # Windows: PowerShell — вся логика
└── install.bat       # Windows: обёртка для двойного клика → запускает install.ps1
```

Вверху каждого скрипта — переменная `REPO_URL`, по умолчанию реальный репозиторий
`https://github.com/tony-kin-dev/TranscribeRU.git`; можно переопределить через
переменную окружения `REPO_URL` (и путь установки — `TRANSCRIBE_RU_DIR`).

## 3. Общая последовательность (оба скрипта)

1. **Пакетный менеджер.** macOS → Homebrew, Windows → winget.
   - Нет → попытка авто-установки (brew: официальный скрипт в режиме
     `NONINTERACTIVE`). Если не вышло без участия пользователя → сообщение со
     ссылкой (https://brew.sh / https://aka.ms/getwinget) и выход с кодом ≠ 0.
2. **Системные зависимости** через менеджер: `python@3.12`, `ffmpeg`, `git`.
3. **Код проекта.** Папка установки: macOS `~/Applications/TranscribeRU`,
   Windows `%LOCALAPPDATA%\TranscribeRU`. Если уже есть `.git` → `git pull`,
   иначе `git clone $REPO_URL`. Идемпотентно (повторный запуск обновляет).
4. **Окружение.** `python3.12 -m venv .venv` в папке проекта → обновить pip →
   `pip install -e .` (тянет `gigaam` из git и остальное).
5. **Иконка на Рабочем столе.**
   - macOS: `.venv/bin/transcribe-ru-gui --install-shortcut` (уже реализовано —
     создаёт `~/Desktop/TranscribeRU.command`).
   - Windows: PowerShell создаёт ярлык `~/Desktop/TranscribeRU.lnk` через
     `WScript.Shell`, цель — `.venv\Scripts\pythonw.exe -m transcribe_ru.gui`
     (`pythonw` — без чёрного окна консоли), рабочая папка — папка проекта.
6. Финал: сообщение «Готово — иконка на Рабочем столе».

## 4. Поиск бинарников (важно)

После установки через пакетник `PATH` в текущей сессии может не обновиться.
Скрипты берут бинарники по известным путям, а не из `PATH`:

- macOS: `BREW="$(command -v brew || echo /opt/homebrew/bin/brew)"`,
  `PY="$($BREW --prefix python@3.12)/bin/python3.12"`.
- Windows: после winget искать Python через лаунчер `py -3.12`, иначе
  `$env:LOCALAPPDATA\Programs\Python\Python312\python.exe`.

## 5. Обработка ошибок и ссылки

Каждый внешний шаг обёрнут проверкой результата. При неустранимой авто-проблеме —
печать (Windows: при отсутствии консоли — `MessageBox`) с конкретной ссылкой:

- Homebrew — https://brew.sh
- winget / App Installer — https://aka.ms/getwinget
- ffmpeg (если менеджер недоступен) — https://ffmpeg.org/download.html
- Python — https://www.python.org/downloads/
- git — https://git-scm.com/downloads

Скрипты идемпотентны: повторный запуск пропускает уже установленное.

## 6. Запуск пользователем

- **macOS:** одной командой
  `curl -fsSL <raw-ссылка>/scripts/install.command | bash`,
  либо скачать `install.command` и дважды кликнуть (после `chmod +x` или
  «Открыть» из контекстного меню).
- **Windows:** скачать `install.bat` (и `install.ps1`) и дважды кликнуть
  `install.bat`; либо в PowerShell
  `iwr -useb <raw>/scripts/install.ps1 | iex`.

README получает раздел «Установка в один шаг» с этими командами и напоминанием
заменить `REPO_URL`/raw-ссылки на свой GitHub.

## 7. Тестирование

Системные bootstrap-скрипты нельзя гонять в обычных автотестах (ставят софт,
требуют пароль, меняют машину). Поэтому:

- `install.command` проверяется `bash -n` (синтаксис) и ревью; полная проверка —
  ручной прогон на чистой macOS.
- `install.ps1`/`.bat` — ревью + ручной прогон на чистой Windows (на dev-машине
  нет PowerShell для линта — это зафиксированное ограничение).
- Логика приложения уже покрыта юнит-тестами; установщики её не меняют.

## 8. Вне рамок (YAGNI)

- Linux-установщик.
- Подписанные/нотаризованные `.app`/`.exe`-инсталляторы.
- Авто-обновление приложения (кроме `git pull` при повторном запуске).
- Удаление/деинсталлятор.
