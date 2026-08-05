"""Тесты общей оркестрации transcribe_to_file (используется и CLI, и GUI)."""

from pathlib import Path

from transcribe_ru.core import Segment, TranscriptResult
from transcribe_ru.runner import output_path, transcribe_batch, transcribe_to_file


def test_output_path_defaults_next_to_source():
    assert output_path("/data/voice.opus", "srt", None) == Path("/data/voice.srt")


def test_output_path_in_out_dir():
    assert output_path("/data/voice.opus", "json", "/tmp/out") == Path("/tmp/out/voice.json")


class _FakeEngine:
    name = "gigaam"
    language = "ru"

    def __init__(self, variant):
        self.variant = variant
        self.loaded = None

    def load(self, device):
        self.loaded = device

    def transcribe_segment(self, wav, sr):
        return ""


def _fake_transcribe(audio_path, engine, *, device, granularity="coarse", on_progress=None):
    engine.load(device)
    if on_progress:
        on_progress(1, 1)
    return TranscriptResult(
        segments=[Segment(0.0, 2.0, "привет"), Segment(2.0, 4.0, "мир")],
        language="ru",
    )


def test_transcribe_to_file_writes_result_and_returns_path(tmp_path):
    audio = tmp_path / "rec.opus"
    audio.write_bytes(b"x")
    progress = []

    out = transcribe_to_file(
        str(audio),
        variant="e2e_ctc",
        device="cpu",
        fmt="txt_plain",
        on_progress=lambda d, t: progress.append((d, t)),
        transcribe_fn=_fake_transcribe,
        get_engine_fn=lambda name: _FakeEngine,
        select_device_fn=lambda d: d,
    )

    assert out == tmp_path / "rec.txt"
    assert out.read_text(encoding="utf-8") == "привет мир"
    assert progress[-1] == (1, 1)


def test_transcribe_to_file_resolves_device_and_passes_variant(tmp_path):
    audio = tmp_path / "rec.wav"
    audio.write_bytes(b"x")
    captured = {}

    def transcribe_fn(audio_path, engine, *, device, granularity="coarse", on_progress=None):
        captured["device"] = device
        captured["variant"] = engine.variant
        return TranscriptResult(segments=[Segment(0.0, 1.0, "а")], language="ru")

    transcribe_to_file(
        str(audio),
        variant="rnnt",
        device="auto",
        fmt="txt_timecoded",
        transcribe_fn=transcribe_fn,
        get_engine_fn=lambda name: _FakeEngine,
        select_device_fn=lambda d: "mps" if d == "auto" else d,
    )

    assert captured == {"device": "mps", "variant": "rnnt"}


def test_transcribe_to_file_passes_granularity(tmp_path):
    audio = tmp_path / "rec.opus"
    audio.write_bytes(b"x")
    captured = {}

    def transcribe_fn(audio_path, engine, *, device, granularity="coarse", on_progress=None):
        captured["granularity"] = granularity
        return TranscriptResult(segments=[Segment(0.0, 1.0, "а")], language="ru")

    transcribe_to_file(
        str(audio),
        fmt="srt",
        granularity="fine",
        transcribe_fn=transcribe_fn,
        get_engine_fn=lambda name: _FakeEngine,
        select_device_fn=lambda d: d,
    )

    assert captured == {"granularity": "fine"}


def test_transcribe_to_file_honors_out_dir(tmp_path):
    audio = tmp_path / "rec.opus"
    audio.write_bytes(b"x")
    out_dir = tmp_path / "results"

    out = transcribe_to_file(
        str(audio),
        fmt="srt",
        out_dir=str(out_dir),
        transcribe_fn=_fake_transcribe,
        get_engine_fn=lambda name: _FakeEngine,
        select_device_fn=lambda d: d,
    )

    assert out == out_dir / "rec.srt"
    assert out.exists()


# --- transcribe_batch (пакет: одна загрузка модели, устойчивость к ошибкам) ---


class _CountingEngine(_FakeEngine):
    """Как _FakeEngine, но считает вызовы load — для проверки «грузим один раз»."""

    def __init__(self, variant):
        super().__init__(variant)
        self.load_calls = 0

    def load(self, device):
        self.load_calls += 1
        self.loaded = device


def _capturing_engine_fn(store):
    """get_engine_fn, который запоминает созданные движки в store (для проверок)."""

    def get_engine_fn(name):
        def make(variant):
            eng = _CountingEngine(variant)
            store.append(eng)
            return eng

        return make

    return get_engine_fn


def test_transcribe_batch_loads_engine_once_and_writes_all(tmp_path):
    a = tmp_path / "a.opus"; a.write_bytes(b"x")
    b = tmp_path / "b.opus"; b.write_bytes(b"x")
    store = []

    summary = transcribe_batch(
        [str(a), str(b)],
        fmt="txt_plain",
        transcribe_fn=_fake_transcribe,
        get_engine_fn=_capturing_engine_fn(store),
        select_device_fn=lambda d: "cpu",
    )

    assert len(store) == 1               # движок создан ОДИН раз на весь пакет
    assert store[0].load_calls == 1      # и загружен ровно один раз
    assert len(summary["done"]) == 2
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "привет мир"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "привет мир"


def test_transcribe_batch_skips_existing(tmp_path):
    a = tmp_path / "a.opus"; a.write_bytes(b"x")
    (tmp_path / "a.txt").write_text("уже есть", encoding="utf-8")  # готовый результат
    b = tmp_path / "b.opus"; b.write_bytes(b"x")

    summary = transcribe_batch(
        [str(a), str(b)],
        fmt="txt_plain",
        skip_existing=True,
        transcribe_fn=_fake_transcribe,
        get_engine_fn=lambda name: _CountingEngine,
        select_device_fn=lambda d: "cpu",
    )

    assert summary["skipped"] == ["a.opus"]
    assert [p.name for p in summary["done"]] == ["b.txt"]
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "уже есть"  # не перезаписан


def test_transcribe_batch_continues_after_error(tmp_path):
    a = tmp_path / "a.opus"; a.write_bytes(b"x")
    b = tmp_path / "b.opus"; b.write_bytes(b"x")

    def flaky(path, engine, *, device, granularity="coarse", on_progress=None):
        engine.load(device)
        if Path(path).name == "a.opus":
            raise RuntimeError("битый файл")
        return TranscriptResult(segments=[Segment(0.0, 1.0, "ок")], language="ru")

    summary = transcribe_batch(
        [str(a), str(b)],
        fmt="txt_plain",
        transcribe_fn=flaky,
        get_engine_fn=lambda name: _CountingEngine,
        select_device_fn=lambda d: "cpu",
    )

    assert [name for name, _ in summary["failed"]] == ["a.opus"]
    assert "битый файл" in summary["failed"][0][1]
    assert [p.name for p in summary["done"]] == ["b.txt"]
    assert not (tmp_path / "a.txt").exists()  # упавший файл не создал результат
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "ок"


def test_transcribe_batch_progress_factory_called_per_file(tmp_path):
    a = tmp_path / "a.opus"; a.write_bytes(b"x")
    b = tmp_path / "b.opus"; b.write_bytes(b"x")
    starts, windows = [], []

    def factory(idx, total, path):
        starts.append((idx, total, Path(path).name))
        return lambda d, t: windows.append((idx, d, t))

    transcribe_batch(
        [str(a), str(b)],
        fmt="txt_plain",
        progress_factory=factory,
        transcribe_fn=_fake_transcribe,
        get_engine_fn=lambda name: _CountingEngine,
        select_device_fn=lambda d: "cpu",
    )

    assert starts == [(1, 2, "a.opus"), (2, 2, "b.opus")]
    assert windows == [(1, 1, 1), (2, 1, 1)]  # _fake_transcribe шлёт on_progress(1, 1)
