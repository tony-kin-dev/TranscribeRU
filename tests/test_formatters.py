"""Тесты форматтеров. Сегмент — любой объект с .start/.end/.text (duck typing)."""

import json
from collections import namedtuple

import pytest

from transcribe_ru import formatters

Seg = namedtuple("Seg", "start end text")

SEGMENTS = [
    Seg(0.0, 4.2, "Привет мир"),
    Seg(12.0, 18.7, "Второй сегмент"),
    Seg(3661.0, 3665.0, "Час прошёл"),  # 1:01:01 — проверка часов
]


def test_txt_timecoded_format():
    out = formatters.format_result(SEGMENTS, "txt_timecoded")
    assert out == (
        "00:00:00\nПривет мир\n\n"
        "00:00:12\nВторой сегмент\n\n"
        "01:01:01\nЧас прошёл\n"
    )


def test_txt_plain_joins_text():
    out = formatters.format_result(SEGMENTS, "txt_plain")
    assert out == "Привет мир Второй сегмент Час прошёл"


def test_srt_format():
    out = formatters.format_result(SEGMENTS, "srt")
    assert out.startswith(
        "1\n00:00:00,000 --> 00:00:04,200\nПривет мир\n\n"
        "2\n00:00:12,000 --> 00:00:18,700\nВторой сегмент\n\n"
    )
    assert "3\n01:01:01,000 --> 01:01:05,000\nЧас прошёл" in out


def test_json_includes_segments_and_meta():
    out = formatters.format_result(
        SEGMENTS, "json", meta={"engine": "gigaam", "variant": "e2e_rnnt"}
    )
    data = json.loads(out)
    assert data["engine"] == "gigaam"
    assert data["variant"] == "e2e_rnnt"
    assert data["segments"][0] == {"start": 0.0, "end": 4.2, "text": "Привет мир"}
    assert len(data["segments"]) == 3


def test_extension_for_each_format():
    assert formatters.EXTENSIONS["txt_timecoded"] == ".txt"
    assert formatters.EXTENSIONS["txt_plain"] == ".txt"
    assert formatters.EXTENSIONS["srt"] == ".srt"
    assert formatters.EXTENSIONS["json"] == ".json"


def test_unknown_format_raises():
    with pytest.raises(ValueError):
        formatters.format_result(SEGMENTS, "doc")
