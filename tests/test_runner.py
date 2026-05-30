"""Тесты общей оркестрации transcribe_to_file (используется и CLI, и GUI)."""

from pathlib import Path

from transcribe_ru.core import Segment, TranscriptResult
from transcribe_ru.runner import output_path, transcribe_to_file


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


def _fake_transcribe(audio_path, engine, *, device, on_progress=None):
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

    def transcribe_fn(audio_path, engine, *, device, on_progress=None):
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
