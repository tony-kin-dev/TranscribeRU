@echo off
REM Установщик TranscribeRU для Windows — двойной клик.
REM Должен лежать рядом с install.ps1 (скачайте оба файла из папки scripts).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
echo.
pause
