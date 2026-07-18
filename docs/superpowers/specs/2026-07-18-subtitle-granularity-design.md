# Дизайн: выбор гранулярности сегментов (coarse / fine)

Дата: 2026-07-18

## Задача

Дать пользователю выбор гранулярности сегментов при транскрипции:

- **coarse** (по умолчанию, текущее поведение) — один сегмент на VAD-окно
  (~20–24 с). Для чтения и «сбора смысла».
- **fine** — короткие реплики ~5–7 с через пословные таймстампы GigaAM. Для
  субтитров с точным таймингом.

Выбор доступен и в CLI, и в GUI. `coarse` остаётся дефолтом — **нулевая
регрессия** существующего поведения.

Решения пользователя: целевая максимальная длина реплики — **до ~7 с**
(верх диапазона, стандарт Netflix/EBU); имя параметра — **`granularity`**
(GUI-подпись «Детализация»).

## Контекст

Пословные таймстампы уже доступны в запиненном коммите пакета `gigaam`
(`6e4b027`, метка «Add timestamps support» старше пина):
`model.transcribe(path, word_timestamps=True)` возвращает
`TranscriptionResult(text, words)`, где `words` — список `Word(text, start, end)`.

Ключевые факты (проверено по исходнику установленного пакета):

1. **Времена слов относительны переданного wav** — то есть относительно начала
   VAD-окна. Ядро прибавляет к ним начало окна.
2. **Пунктуация приклеена к слову**: `frames_to_words` группирует токены по
   границам (пробел / префикс `▁`), а знаки `.`, `!`, `?`, `,` отдельным словом
   не выделяются — «Привет,», «как», «дела?». Конец предложения ловится по
   последнему символу слова.
3. **`word_timestamps=True` не активирует longform/pyannote.** `transcribe()`
   при длине ≤ порога (25 с) идёт через `prepare_wav → forward → _decode` без
   `transcribe_longform`; pyannote нужен только как установленный пакет (уже
   учтено), gated-модель и `HF_TOKEN` не требуются. Наши окна ≤ 24 с.
4. **Варианты без пунктуации** (`rnnt`, `ctc` без `e2e`) вернут слова без знаков
   препинания — нарезка «по предложению» там не сработает, нужна деградация.
5. `str(TranscriptionResult) == result.text`, поэтому текущий путь движка
   остаётся эквивалентным прежнему.

Форматтеры ([formatters.py](../../../src/transcribe_ru/formatters.py)) работают
с любым объектом, у которого есть `.start` / `.end` / `.text` (duck-typing),
поэтому их менять не нужно.

## Архитектура

Слова протягиваются **вторым необязательным каналом** — контракт
`transcribe_segment(wav, sr) -> str` не меняется.

### 1. Интерфейс движка — `engines/base.py`

Добавить в ABC `Engine` **не-абстрактный** метод с дефолтом:

```python
def transcribe_words(self, wav, sr):
    """Слова с временами ОТНОСИТЕЛЬНО начала окна (объекты с .text/.start/.end),
    либо None, если движок не умеет пословные таймстампы."""
    return None
```

Не-абстрактный ⇒ существующие движки и тестовые фейки менять не нужно. `None`
даёт бесплатную мягкую деградацию: незнакомый движок в режиме `fine` не падает,
а откатывается на `coarse`-ветку.

### 2. Движок GigaAM — `engines/gigaam.py`

Вынести повторяющийся «танец» с временным wav в приватный `_run`, чтобы не
дублировать, и добавить `transcribe_words`:

```python
def _run(self, wav, sr, **kw):
    if self._model is None:
        raise RuntimeError("Движок не загружен; сначала вызовите load(device).")
    path = _write_temp_wav(wav, sr)
    try:
        return self._model.transcribe(path, **kw)
    finally:
        os.unlink(path)

def transcribe_segment(self, wav, sr):
    return str(self._run(wav, sr)).strip()

def transcribe_words(self, wav, sr):
    return getattr(self._run(wav, sr, word_timestamps=True), "words", None)
```

`word_timestamps=True` вызывается **только** на `fine`-пути; дефолтный `coarse`
за выравнивание не платит. Старый путь (без kwargs) эквивалентен прежнему
`self._model.transcribe(path)` — тесты gigaam с `FakeModel.transcribe(path)`
проходят без правок.

### 3. Ресегментация — `group_words()` в `core.py`

Одна чистая функция рядом с `Segment` (0 новых файлов, `gigaam` не
импортируется, duck-typing по `.text`/`.start`/`.end`):

```python
def group_words(words, *, offset=0.0, max_dur=7.0, gap=0.6, max_chars=84,
                min_dur=1.2, min_chars=6):
    ...
```

Параметры — **калибровочные дефолты функции**, в CLI/GUI не выносятся (YAGNI;
вынести, если реально попросят тюнинг).

- `max_dur=7.0` — верх целевого диапазона 5–7 с (Netflix max).
- `gap=0.6` — пауза, по которой делим реплики.
- `max_chars=84` — две строки субтитра по ~42 символа (EBU/Netflix RU).
- `min_dur=1.2` / `min_chars=6` — **анти-мигание**: финальный проход склеивает
  короткие вставки («Угу», «Да», «И» — реплика короче `min_dur` И с текстом
  ≤ `min_chars`) с соседней, чтобы они не выводились отдельным мелькающим
  тайм-кодом. Добавлено 2026-07-18 после теста на реальной записи 1:10, где
  fine-режим давал десятки односекундных вставок (изначально порог был убран как
  YAGNI, но мигание подтвердилось на живых данных).

**Алгоритм** (жадная склейка). Слова окна имеют времена относительно окна;
`offset` (= начало окна) прибавляется при выпуске реплики. Функция вызывается
**поокно** — реплика не пересекает границу VAD-окна, буфер флашится в конце окна.

Аккумулируем слова в `buf`. Перед добавлением слова `w` флашим `buf`, если
срабатывает любой жёсткий предел:

- `w.end - buf[0].start > max_dur` — реплика длиннее ~7 с;
- `len(текущий_текст) + 1 + len(w.text) > max_chars` — длиннее двух строк;
- `w.start - buf[-1].end > gap` — пауза больше 0.6 с (реальный разрыв: внутри
  склеенного VAD-окна `pack_windows` мержит интервалы через паузы).

Затем `buf.append(w)`. После добавления — мягкий разрыв по концу предложения:
если `w.text` непустой и `w.text[-1] in ".!?…"`, флашим (guard на непустой
`w.text` — против `IndexError`). В конце — финальный флаш остатка.

Выпуск реплики: `Segment(buf[0].start + offset, buf[-1].end + offset,
" ".join(w.text for w in buf).strip())`; пустой текст пропускаем. Первое слово в
буфере никогда не режем — одиночное слово длиннее `max_dur`/`max_chars` выходит
целой репликой (слово не дробим).

**Деградация без пунктуации** (`rnnt`/`ctc`): в словах нет `.!?…` ⇒ мягкий
разрыв не срабатывает никогда, остаются жёсткие пределы (пауза, `max_dur`,
`max_chars`) ⇒ ровные реплики ~5–7 с, без падений.

### 4. Ядро — `core.py`

`transcribe(...)` получает параметр `granularity="coarse"`. В цикле по окнам:

```python
words = engine.transcribe_words(chunk, sr) if granularity == "fine" else None
if words:
    segments.extend(group_words(words, offset=start))
else:
    text = engine.transcribe_segment(chunk, sr).strip()
    if text:
        segments.append(Segment(start, end, text))
```

При `coarse` (дефолт) `words` всегда `None` ⇒ ветка `else` побайтово равна
текущему коду ⇒ нулевая регрессия. В `fine` при непустых словах
`transcribe_segment` не вызывается ⇒ нет двойного инференса.

Тихое/пустое окно в `fine`: `transcribe_words` вернёт `[]`/`None` ⇒ откат на
`transcribe_segment` (второй прогон модели только на пустых окнах — отметить
`# ponytail:`-комментарием, это редкий случай). `on_progress` по-прежнему
вызывается раз на окно.

### 5. Проброс параметра — `runner.py`, `cli.py`, `gui.py`

- `runner.transcribe_to_file(..., granularity="coarse")` → передаёт
  `granularity=granularity` в инъектируемый `transcribe_fn`.
- `cli.py`: рядом с `FORMATS` объявить `GRANULARITIES = ("coarse", "fine")`.
  Флаг `--granularity` (`default="coarse"`, `choices=GRANULARITIES`,
  help: «coarse — крупные окна ~20–24 с (по умолчанию, для чтения); fine —
  короткие реплики ~5–7 с для субтитров»). `main()` прокидывает
  `granularity=args.granularity`. В `dry-run` — строка «гранулярность=…».
- `gui.py`: добавить
  ```python
  GRANULARITY_CHOICES = [
      ("Крупные фрагменты (для чтения)", "coarse"),   # первый = дефолт
      ("Короткие реплики (для субтитров)", "fine"),
  ]
  ```
  `StringVar(value=GRANULARITY_CHOICES[0][0])`; новый `_combo` с подписью
  «Детализация:» после «Формат:» (перенумеровать `row` у variant/device/кнопки/
  прогресс-бара/статуса на +1); в `params` —
  `granularity=label_to_value(GRANULARITY_CHOICES, ...)`; проброс в `_run_job`.
  Тот же механизм `label_to_value` — новых сущностей нет.

### 6. Форматтеры и выходные форматы

Форматтеры **не трогаем**. `fine` выигрывает для `srt` (короткие реплики с
точными `-->` таймингами), `json` (мелкие `start`/`end`), `txt_timecoded`
(частые метки времени). `txt_plain` склеивает только `.text` — при `fine` это те
же слова, разбитые иначе, содержание практически то же (тайминги игнорируются),
спецобработки не требует. Перенос строки (≤42 символа) в текст реплики **не**
вставляем — это отдано плееру; иначе пришлось бы трогать форматтер и пачкать
`txt_plain`.

## Изменяемые файлы

Новых модулей нет.

| Файл | Изменение |
|------|-----------|
| `src/transcribe_ru/engines/base.py` | +не-абстрактный `transcribe_words -> None` |
| `src/transcribe_ru/engines/gigaam.py` | рефактор в `_run(**kw)` + `transcribe_words` |
| `src/transcribe_ru/core.py` | +`group_words(...)`; `transcribe(..., granularity="coarse")` + ветвление |
| `src/transcribe_ru/runner.py` | `transcribe_to_file(..., granularity="coarse")`, проброс |
| `src/transcribe_ru/cli.py` | `GRANULARITIES`; флаг `--granularity`; проброс; строка в dry-run |
| `src/transcribe_ru/gui.py` | `GRANULARITY_CHOICES`; `StringVar`; +один `_combo`; проброс; перенумерация grid |
| `tests/test_core.py` | +тест `fine`-нарезки (DI-движок) + микро-assert деградации `group_words` |
| `tests/test_runner.py` | 2 фейка `transcribe_fn`: +kwarg `granularity="coarse"` |
| `tests/test_cli.py` | `_fake_transcribe`: +kwarg `granularity="coarse"` |
| `tests/test_gui_choices.py` | покрытие `GRANULARITY_CHOICES` (значения == `cli.GRANULARITIES`, дефолт первый) |

## Граничные случаи

- `coarse` (дефолт): `words=None` ⇒ ветка идентична текущему коду, нулевая
  регрессия.
- Вариант без пунктуации (`rnnt`/`ctc`): деградация на паузы + `max_dur` +
  `max_chars`, реплики ~5–7 с.
- Тишина/пустое окно: `transcribe_words -> []`/`None` ⇒ откат на
  `transcribe_segment`; пустой текст `.strip()` не добавляется.
- Незнакомый движок без `transcribe_words` в `fine`: дефолт ABC → `None` →
  корректный откат на `coarse`, без `AttributeError`.
- Пустой `w.text`: guard `if w.text and w.text[-1] in ...` против `IndexError`.
- Времена слов относительны окна ⇒ ядро прибавляет `offset = start` окна
  (проверяется тестом с ненулевым стартом 10.0).
- Пауза внутри склеенного VAD-окна ⇒ `gap` делит реплики по реальным паузам.
- Одиночное слово длиннее `max_dur`/`max_chars` ⇒ выходит целой репликой (первое
  слово не режем).
- Предложение на стыке двух VAD-окон: буфер флашится в конце окна — фраза может
  разорваться границей окна. Для ~20–24 с окон, режущихся по тишине, редко;
  принятое ограничение.

## План тестирования

Один самодостаточный `pytest` без torch/модели в `tests/test_core.py` через
инъекцию `load_audio`/`segment_fn` + DI-движок с `transcribe_words`:

```python
class _W:
    def __init__(self, text, start, end):
        self.text, self.start, self.end = text, start, end

class FineEngine(FakeEngine):  # наследует load/transcribe_segment
    def transcribe_words(self, wav, sr):
        return [_W("Привет,", 0.0, 0.5), _W("мир.", 0.6, 1.6),
                _W("Как", 1.7, 2.0), _W("дела?", 2.1, 2.8)]

def test_fine_splits_words_into_cues_with_offset():
    r = transcribe("d.opus", FineEngine(), device="cpu", granularity="fine",
                   load_audio=lambda p: (["full"], 16000),
                   segment_fn=lambda w, s: [(["w"], (10.0, 30.0))])  # окно с 10.0
    assert [(round(s.start, 1), round(s.end, 1), s.text) for s in r.segments] == [
        (10.0, 11.6, "Привет, мир."),   # разрыв по концу предложения, offset +10.0
        (11.7, 12.8, "Как дела?")]

def test_group_words_degrades_without_punctuation():
    from transcribe_ru.core import group_words
    cues = group_words([_W("раз", 0, 1), _W("два", 4, 5)], gap=0.6)  # пауза 3с > gap
    assert len(cues) == 2  # без пунктуации режем по паузе
```

Покрывает: проброс `granularity`, вызов `transcribe_words` вместо
`transcribe_segment`, группировку по концу предложения + паузе, склейку слов,
прибавление `offset`, деградацию без пунктуации. Существующие `coarse`-тесты — как
есть; правятся только 3 DI-фейка `transcribe_fn` (+kwarg `granularity="coarse"`).

Ручная проверка: реальный прогон `transcribe-ru --audio <файл> --granularity fine
--format srt` на видео/аудио с речью — убедиться, что реплики ~5–7 с, пунктуация
и тайминги на месте (в т.ч. что `word_timestamps=True` не тянет pyannote в
рантайме).

## Объём

~1 день, зависимости не добавляются, 6 файлов `src` + 4 файла тестов, ноль новых
модулей.
