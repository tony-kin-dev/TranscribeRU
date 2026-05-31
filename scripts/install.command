#!/bin/bash
# Установщик TranscribeRU для macOS: ставит Homebrew/Python/ffmpeg/git,
# клонирует проект, создаёт venv, ставит зависимости и кладёт иконку на стол.
# Запуск: двойной клик, либо
#   curl -fsSL https://raw.githubusercontent.com/tony-kin-dev/TranscribeRU/main/scripts/install.command | bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/tony-kin-dev/TranscribeRU.git}"
DEST="${TRANSCRIBE_RU_DIR:-$HOME/Applications/TranscribeRU}"

say() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
fail() { printf '\n\033[31m%s\033[0m\n' "$1" >&2; exit 1; }

# 1. Homebrew -------------------------------------------------------------
if ! command -v brew >/dev/null 2>&1; then
  say "Homebrew не найден — устанавливаю (может потребоваться пароль)..."
  if ! NONINTERACTIVE=1 /bin/bash -c \
      "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"; then
    fail "Не удалось установить Homebrew автоматически.
Установите его вручную (инструкция на https://brew.sh), затем запустите скрипт снова."
  fi
fi
# brew может быть не в PATH этой сессии — ищем по стандартным путям
BREW="$(command -v brew || true)"
[ -z "$BREW" ] && [ -x /opt/homebrew/bin/brew ] && BREW=/opt/homebrew/bin/brew
[ -z "$BREW" ] && [ -x /usr/local/bin/brew ] && BREW=/usr/local/bin/brew
[ -z "$BREW" ] && fail "Homebrew установлен, но не найден. Перезапустите терминал и скрипт."

# 2. Системные зависимости -----------------------------------------------
say "Устанавливаю Python 3.12, ffmpeg и git через Homebrew..."
"$BREW" install python@3.12 ffmpeg git || fail \
  "Не удалось установить зависимости через Homebrew. Проверьте интернет и запустите снова."

PY="$("$BREW" --prefix python@3.12)/bin/python3.12"
[ -x "$PY" ] || PY="$(command -v python3.12 || true)"
[ -x "$PY" ] || fail "Python 3.12 не найден после установки. Скачайте с https://www.python.org/downloads/"

# 3. Код проекта ----------------------------------------------------------
if [ -d "$DEST/.git" ]; then
  say "Обновляю существующую установку в $DEST..."
  git -C "$DEST" pull --ff-only || true
else
  say "Скачиваю проект в $DEST..."
  mkdir -p "$(dirname "$DEST")"
  git clone "$REPO_URL" "$DEST" || fail "Не удалось клонировать $REPO_URL"
fi

# 4. Виртуальное окружение и зависимости ----------------------------------
say "Создаю окружение и ставлю зависимости (первый раз — несколько минут)..."
"$PY" -m venv "$DEST/.venv"
"$DEST/.venv/bin/python" -m pip install --quiet --upgrade pip
"$DEST/.venv/bin/python" -m pip install -e "$DEST" || fail \
  "Не удалось установить зависимости. Проверьте интернет и запустите снова."

# 5. Иконка на Рабочем столе ----------------------------------------------
say "Создаю иконку на Рабочем столе..."
"$DEST/.venv/bin/transcribe-ru-gui" --install-shortcut

say "Готово! Иконка TranscribeRU.command на Рабочем столе — двойной клик запускает окно."
