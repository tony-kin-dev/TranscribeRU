# TranscribeRU

Кроссплатформенная CLI-платформа для распознавания речи (ASR) на PyTorch-движках.
Первый движок — **GigaAM-v3** (`ai-sage/GigaAM-v3`), Conformer/RNN-T от Salute
с заметно лучшим качеством на русском, чем Whisper-large-v3.

Архитектура движков-плагинов: ядро владеет нарезкой длинного аудио и сборкой
сегментов, движок отвечает только за «звук одного окна → текст». Добавить новый
движок = один файл в `engines/` + запись в реестр.

## Возможности

- Любой входной формат через системный `ffmpeg` (`.opus/.m4a/.wav/.mp3/…`).
- Нарезка длинных файлов через `silero-vad` (MIT, без HF-токена).
- Авто-выбор устройства: CUDA → MPS → CPU.
- Форматы вывода: `txt_timecoded`, `txt_plain`, `srt`, `json`.

## Установка

Требуется Python **3.11/3.12** (пакет `gigaam` пинит `torch<=2.5.1`/`onnxruntime`,
под 3.13+ колёс нет) и системный `ffmpeg`.

```bash
uv sync            # или: pip install -e .
```

## Использование

```bash
transcribe-ru --audio file.opus
transcribe-ru --audio file.opus --variant e2e_ctc --device cpu --out-dir ./out
transcribe-ru --audio file.opus --format srt
```

| Флаг | По умолчанию | Описание |
|------|--------------|----------|
| `--audio` | — (обяз.) | Входной аудиофайл |
| `--engine` | `gigaam` | Движок ASR |
| `--variant` | `e2e_rnnt` | Вариант модели: `e2e_rnnt`, `e2e_ctc`, `rnnt`, `ctc` |
| `--device` | `auto` | `auto`, `cuda`, `mps`, `cpu` |
| `--format` | `txt_timecoded` | `txt_timecoded`, `txt_plain`, `srt`, `json` |
| `--out-dir` | рядом с исходником | Каталог для результата |
| `--dry-run` | — | Показать план без запуска |

## Разработка

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pytest                 # юнит-тесты (быстрые)
pytest -m slow         # интеграционный тест GigaAM (скачивает модель)
```

## Лицензия

MIT
