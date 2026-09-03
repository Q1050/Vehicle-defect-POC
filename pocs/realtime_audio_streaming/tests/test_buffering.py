import numpy as np

from realtime_audio.buffering import RollingAudioBuffer
from realtime_audio.encoding import float32_to_pcm_s16le, pcm_s16le_to_float32


def test_pcm_round_trip(sine_samples):
    restored = pcm_s16le_to_float32(float32_to_pcm_s16le(sine_samples))
    assert restored.dtype == np.float32
    assert np.max(np.abs(restored - sine_samples)) < 1e-4


def test_four_second_window_and_one_second_hop():
    buffer = RollingAudioBuffer(10, window_seconds=4, hop_seconds=1)
    buffer.append(np.arange(60, dtype=np.float32))
    windows = buffer.pop_available_windows()
    assert [(item.start_seconds, item.end_seconds) for item in windows] == [
        (0, 4), (1, 5), (2, 6)
    ]
    assert np.array_equal(windows[1].samples, np.arange(10, 50, dtype=np.float32))


def test_stale_samples_are_discarded_but_overlap_retained():
    buffer = RollingAudioBuffer(10, window_seconds=4, hop_seconds=1)
    buffer.append(np.ones(40, dtype=np.float32))
    assert len(buffer.pop_available_windows()) == 1
    assert buffer.buffered_samples == 30
    buffer.append(np.ones(10, dtype=np.float32))
    second = buffer.pop_available_windows()
    assert len(second) == 1
    assert second[0].start_seconds == 1
    assert buffer.buffered_samples == 30
