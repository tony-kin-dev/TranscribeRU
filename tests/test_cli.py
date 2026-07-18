"""Тесты CLI: парсинг аргументов, путь вывода и оркестрация main (с инъекцией)."""

from pathlib import Path

import pytest

from transcribe_ru import cli
from transcribe_ru.core import Segment, TranscriptResult


def test_parser_defaults():
    args = cli.build_parser().parse_args(["--audio", "f.opus"])
    assert args.audio == "f.opus"
    assert args.engine == "gigaam"
    assert args.variant == "e2e_rnnt"
    assert args.device == "auto"
    assert args.format == "txt_timecoded"
    assert args.granularity == "coarse"
    assert args.out_dir is None
    assert args.dry_run is False


def test_parser_rejects_unknown_granularity():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--audio", "f.opus", "--granularity", "medium"])


def test_parser_requires_audio():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([])


def test_parser_rejects_unknown_format():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--audio", "f.opus", "--format", "doc"])


def test_output_path_defaults_next_to_source():
    p = cli.output_path("/data/voice.opus", "srt", None)
    assert p == Path("/data/voice.srt")


def test_output_path_in_out_dir_with_format_extension():
    p = cli.output_path("/data/voice.opus", "json", "/tmp/out")
    assert p == Path("/tmp/out/voice.json")


def _fake_transcribe(audio_path, engine, *, device, granularity="coarse", on_progress=None):
    if on_progress:
        on_progress(1, 1)
    return TranscriptResult(
        segments=[Segment(0.0, 2.0, "привет"), Segment(2.0, 4.0, "мир")],
        language="ru",
    )


class _FakeEngine:
    name = "gigaam"
    language = "ru"

    def __init__(self, variant):
        self.variant = variant

    def load(self, device):
        pass

    def transcribe_segment(self, wav, sr):
        return ""


def test_main_writes_formatted_output(tmp_path, capsys):
    audio = tmp_path / "rec.opus"
    audio.write_bytes(b"fake")

    rc = cli.main(
        ["--audio", str(audio), "--format", "txt_plain", "--device", "cpu"],
        transcribe_fn=_fake_transcribe,
        get_engine_fn=lambda name: _FakeEngine,
        select_device_fn=lambda d: d,
    )

    assert rc == 0
    out_file = tmp_path / "rec.txt"
    assert out_file.read_text(encoding="utf-8") == "привет мир"


def test_main_dry_run_does_not_transcribe(tmp_path):
    audio = tmp_path / "rec.opus"
    audio.write_bytes(b"fake")
    called = []

    rc = cli.main(
        ["--audio", str(audio), "--dry-run"],
        transcribe_fn=lambda *a, **k: called.append(1),
        get_engine_fn=lambda name: _FakeEngine,
        select_device_fn=lambda d: "cpu",
    )

    assert rc == 0
    assert called == []
    assert not (tmp_path / "rec.txt").exists()
