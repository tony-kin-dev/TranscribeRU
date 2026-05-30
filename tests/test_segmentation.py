"""Тесты нарезки. VAD инъектируется; проверяем чистую логику окон и резку wav."""

import numpy as np

from transcribe_ru.segmentation import pack_windows, segment, slice_audio

MAX = 24.0
TARGET = 20.0


def _within_limit(windows, limit=MAX):
    return all((e - s) <= limit + 1e-9 for s, e in windows)


def _monotonic(windows):
    last = -1.0
    for s, e in windows:
        if s < last or e <= s:
            return False
        last = e
    return True


def test_short_adjacent_intervals_merge_into_one_window():
    intervals = [(0.0, 3.0), (3.5, 6.0), (6.2, 9.0)]
    windows = pack_windows(intervals, max_len=MAX, target_len=TARGET)
    assert windows == [(0.0, 9.0)]


def test_merge_stops_at_target_length():
    # три интервала по 8с с зазорами — суммарно превышают target 20с
    intervals = [(0.0, 8.0), (8.0, 16.0), (16.0, 24.0)]
    windows = pack_windows(intervals, max_len=MAX, target_len=TARGET)
    assert len(windows) == 2
    assert _within_limit(windows)
    assert _monotonic(windows)


def test_long_interval_is_split_below_max():
    intervals = [(0.0, 60.0)]  # длиннее max — режется на части
    windows = pack_windows(intervals, max_len=MAX, target_len=TARGET)
    assert len(windows) >= 3
    assert _within_limit(windows)
    assert _monotonic(windows)
    # покрытие непрерывно от начала до конца
    assert windows[0][0] == 0.0
    assert windows[-1][1] == 60.0


def test_empty_intervals_give_no_windows():
    assert pack_windows([], max_len=MAX, target_len=TARGET) == []


def test_slice_audio_cuts_chunks_at_sample_boundaries():
    sr = 16000
    wav = np.arange(sr * 10, dtype=np.float32)  # 10с
    windows = [(0.0, 4.0), (4.0, 10.0)]
    chunks = slice_audio(wav, sr, windows)
    assert len(chunks) == 2
    chunk0, (s0, e0) = chunks[0]
    assert (s0, e0) == (0.0, 4.0)
    assert len(chunk0) == 4 * sr
    assert chunk0[0] == 0.0
    chunk1, (s1, e1) = chunks[1]
    assert len(chunk1) == 6 * sr


def test_segment_uses_injected_vad_and_respects_limit():
    sr = 16000
    wav = np.zeros(sr * 60, dtype=np.float32)
    fake_speech = lambda w, s: [(0.0, 60.0)]
    chunks = segment(wav, sr, max_len=MAX, target_len=TARGET, get_speech=fake_speech)
    windows = [bounds for _, bounds in chunks]
    assert _within_limit(windows)
    assert _monotonic(windows)
    # каждому окну соответствует непустой кусок звука
    assert all(len(chunk) > 0 for chunk, _ in chunks)


def test_segment_without_speech_falls_back_to_whole_file():
    sr = 16000
    wav = np.zeros(sr * 5, dtype=np.float32)
    chunks = segment(wav, sr, max_len=MAX, target_len=TARGET, get_speech=lambda w, s: [])
    assert len(chunks) == 1
    _, (start, end) = chunks[0]
    assert start == 0.0
    assert abs(end - 5.0) < 1e-6
