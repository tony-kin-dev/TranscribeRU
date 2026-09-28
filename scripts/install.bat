@echo off
REM Установщик TranscribeRU для Windows — двойной клик.
REM Должен лежать рядом с install.ps1 (скачайте оба файла из папки scripts).
REM Не -File: Windows PowerShell 5.1 читает .ps1 без BOM в ANSI (cp1251), и
REM кириллица/тире превращаются в «умные кавычки» → скрипт не парсится вовсе.
REM Читаем явно как UTF-8. Путь — через переменную, чтобы не ломали апострофы.
set "TRU_PS1=%~dp0install.ps1"
powershell -NoProfile -ExecutionPolicy Bypass -Command "iex ([IO.File]::ReadAllText($env:TRU_PS1))"
echo.
pause
