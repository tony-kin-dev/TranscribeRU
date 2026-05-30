# TranscribeRU — дизайн

Дата: 2026-05-30
Статус: утверждён к реализации

## 1. Назначение и контекст

Кроссплатформенная CLI-платформа для распознавания речи (ASR) на PyTorch-движках,
с заделом под перевод. Первый движок — **GigaAM-v3** (`ai-sage/GigaAM-v3`),
Conformer/RNN-T-модель от Salute с заметно лучшим качеством на русском, чем
Whisper-large-v3 (средний WER ~8.4% против ~25% на их бенчмарках).

Это **отдельный проект**, не связанный с `whisper-transcribe`. Причина:
`whisper-transcribe` жёстко завязан на `whisperkit-cli`/CoreML (только macOS/Apple
Silicon) и понимает только архитектуру Whisper. GigaAM — другая архитектура, через
WhisperKit не запускается. Чистый PyTorch (через официальный пакет `gigaam`) даёт
кроссплатформенность (Linux/macOS/Windows) сам по себе.

### Решения, зафиксированные на брейншторминге
- **Рамки:** платформа под много моделей (движки-плагины), но на старте — только ASR.
  Перевод — задел на будущее, в первой версии не кодим.
- **Интерфейс:** только CLI. Без GUI/веб/Docker (Docker на Mac теряет GPU/MPS —
  для пользователя это был бы downgrade).
- **Кроссплатформенность:** через чистый Python + `uv`/`pip`. Ничего
  macOS-специфичного.
- **Расположение:** новый отдельный git-репо `~/Projects/TranscribeRU`.
- **Нарезка длинных файлов:** `silero-vad` (MIT, без токена и gating), а не штатный
  `pyannote/segmentation-3.0` (gated, требует HF-токен).

## 2. Архитектура и раскладка

```
TranscribeRU/
├── pyproject.toml          # uv/pip, зависимости, console_script «transcribe-ru»
├── README.md
├── src/transcribe_ru/
│   ├── core.py             # Segment, TranscriptResult; оркестрация нарезка→движок
│   ├── engines/
│   │   ├── base.py         # ABC Engine + реестр ENGINES
│   │   └── gigaam.py       # движок GigaAM-v3 (PyTorch)
│   ├── segmentation.py     # silero-vad: аудио → список окон <25с с границами
│   ├── audio.py            # загрузка/ресемпл через ffmpeg → 16 kHz mono float32
│   ├── formatters.py       # Segment[] → txt_timecoded / txt_plain / srt / json
│   ├── device.py           # авто-выбор устройства: cuda > mps > cpu
│   └── cli.py              # argparse, прогресс в stderr, запись результата
└── tests/
```

### Принципы
- Ядро не знает о конкретном движке — общается через `Engine`-интерфейс.
  Новый движок = один файл в `engines/` + запись в реестр.
- Ядро владеет нарезкой и сборкой сегментов; движок отвечает только за
  «звук одного окна → текст».
- Формат вывода с тайм-кодами совместим с `whisper-transcribe`
  (`00:00:12\nтекст\n\n00:00:18\n…`).

## 3. Интерфейс движков (ядро платформы)

```python
class Engine(ABC):
    name: str
    def load(self, device: str) -> None: ...           # ленивая загрузка модели
    def transcribe_segment(self, wav, sr: int) -> str:  # один <25с кусок → текст
        ...

ENGINES: dict[str, type[Engine]] = {"gigaam": GigaAMEngine}
```

`core.py`:
- `Segment(start: float, end: float, text: str)` (frozen dataclass).
- `TranscriptResult(segments: list[Segment], language: str | None)`.
- `transcribe(audio_path, engine, *, device, on_progress) -> TranscriptResult`:
  1. `audio.load()` → 16 kHz mono float32;
  2. `segmentation.segment()` → `[(wav_chunk, (start, end)), …]`, каждый <25с;
  3. цикл: `engine.transcribe_segment(chunk)` → текст; собираем `Segment`;
  4. прогресс через `on_progress(done, total, ...)`.

## 4. Движок GigaAM (`engines/gigaam.py`)

> **Уточнение по факту интеграции (2026-05-30).** Изначально планировался маршрут
> `transformers.AutoModel.from_pretrained("ai-sage/GigaAM-v3", trust_remote_code=True)`.
> Интеграционный (`@slow`) тест вскрыл две проблемы этого маршрута: (1) transformers
> сканирует импорты `modeling_gigaam.py` и **требует `pyannote` в окружении при любой
> загрузке** (хотя реально pyannote нужен лишь в longform); (2) `model.transcribe()`
> принимает **путь к файлу, не массив**. Поэтому перешли на официальный пакет
> **`gigaam`** (Salute), где pyannote вынесен в extra `gigaam[longform]`.

- Загрузка: `gigaam.load_model("v3_<variant>", device=device)`.
- Вариант по умолчанию — **`e2e_rnnt`** (пунктуация + нормализация из коробки).
  Доступны: `e2e_rnnt`, `e2e_ctc`, `rnnt`, `ctc` (через `--variant`); движок
  маппит их в имена пакета `v3_e2e_rnnt`, `v3_e2e_ctc`, `v3_rnnt`, `v3_ctc`.
- `transcribe_segment` получает numpy-окно <25с (ниже `LONGFORM_THRESHOLD = 25с`),
  пишет его во временный 16 кГц mono wav и зовёт публичный `model.transcribe(path)`
  (вариант B — устойчив к версиям, не использует внутренние атрибуты модели).
  `transcribe_longform` не используем — нарезку длинных файлов делает silero-vad
  в ядре, поэтому gated `pyannote/segmentation-3.0` и `HF_TOKEN` не нужны.
- Зависимость движка — пакет `gigaam` (тянет `torch<=2.5.1`, `torchaudio`,
  `hydra-core`, `omegaconf`, `sentencepiece`, `onnx`/`onnxruntime`). Из-за пинов
  пакета требуется **Python 3.11/3.12** (на 3.13+ нет колёс).

## 5. Нарезка (`segmentation.py`)

- `silero-vad` определяет интервалы речи; склеиваем/режем под лимит <25с
  (с небольшим запасом, напр. целевое окно ~20с), границы тишины — точки реза.
- Возвращает `[(wav_chunk_float32, (start_sec, end_sec)), …]`.
- Границы окон становятся тайм-кодами сегментов на выходе.

## 6. Загрузка аудио (`audio.py`)

- Через системный `ffmpeg` (как делает сам GigaAM): любой вход
  (`.opus/.m4a/.wav/.mp3/…`) → PCM s16le → float32, 16 kHz, моно.
- Если `ffmpeg` не найден — понятная ошибка с подсказкой по установке.

## 7. CLI (`cli.py`)

```bash
transcribe-ru --audio file.opus
transcribe-ru --audio file.opus --variant e2e_ctc --device cpu --out-dir ./out
transcribe-ru --audio file.opus --format srt
```

Флаги: `--audio` (обяз.), `--engine` (дефолт `gigaam`), `--variant`
(дефолт `e2e_rnnt`), `--device` (дефолт `auto`), `--format`
(`txt_timecoded`|`txt_plain`|`srt`|`json`, дефолт `txt_timecoded`),
`--out-dir` (дефолт — рядом с исходником), `--dry-run`.

Прогресс — в stderr: готово сегментов / всего, %, ETA. Результат — файл рядом
с исходником с тем же именем и расширением по формату.

## 8. Форматтеры (`formatters.py`)

- `txt_timecoded`: `HH:MM:SS\nтекст\n\n…` (как в `whisper-transcribe`).
- `txt_plain`: только текст, склеенный по сегментам.
- `srt`: стандартные субтитры с индексами и `-->`-таймкодами.
- `json`: сырой `[{start, end, text}, …]` + метаданные (движок, вариант).

## 9. Перевод (будущее, не реализуем сейчас)

Тот же паттерн движков: `Translator`-плагин (например на NLLB/seamless) с
интерфейсом `translate(text, src, tgt) -> str`, отдельный файл в `engines/`.
В первой версии не регистрируем и не кодим. Решение фиксируем, чтобы интерфейс
ядра не пришлось переделывать.

## 10. Зависимости и окружение

- Менеджер: `uv` (или `pip`). Python **3.11/3.12** (пакет `gigaam` пинит
  `torch<=2.5.1`/`onnxruntime`, под 3.13+ колёс нет).
- Пакеты: `gigaam` (тянет `torch`, `torchaudio`, `sentencepiece`, `hydra-core`,
  `omegaconf`, `onnx`/`onnxruntime`), `silero-vad`. pyannote — НЕ ставим
  (он только в extra `gigaam[longform]`, который не используем).
- Системно: `ffmpeg`.
- Ускорение: CUDA (Linux/Windows), MPS (macOS Apple Silicon), CPU-фолбэк.

## 11. Тестирование

- `pytest`. Юниты без скачивания модели:
  - `segmentation`: на синтетическом сигнале (тишина/речь-заглушка) проверяем,
    что окна не длиннее лимита и границы монотонны;
  - `formatters`: `Segment[]` → каждый формат, проверка тайм-кодов и структуры;
  - `device`: выбор устройства при разных доступностях (моки `torch`);
  - `core`: оркестрация с фейковым `Engine` (без torch) — нарезка→сборка.
- Движок GigaAM — отдельный медленный интеграционный тест (помечен `@slow`,
  пропускается по умолчанию; реально скачивает модель и гоняет на коротком wav).

## 12. Вне рамок (YAGNI)

- GUI / веб / Docker / `.app`-бандл.
- Перевод (только интерфейсный задел).
- Дообучение/конверсия моделей.
- Диаризация (кто говорит).
