"""Bounded overlapping-window PCM sample buffer."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AudioWindow:
    samples: np.ndarray
    start_sample: int
    end_sample: int
    sample_rate: int

    @property
    def start_seconds(self) -> float:
        return self.start_sample / self.sample_rate

    @property
    def end_seconds(self) -> float:
        return self.end_sample / self.sample_rate


class RollingAudioBuffer:
    def __init__(self, sample_rate: int, window_seconds: float, hop_seconds: float):
        self.sample_rate = sample_rate
        self.window_samples = round(window_seconds * sample_rate)
        self.hop_samples = round(hop_seconds * sample_rate)
        if self.window_samples <= 0 or self.hop_samples <= 0:
            raise ValueError("Window and hop sizes must be positive.")
        if self.hop_samples > self.window_samples:
            raise ValueError("Hop size must not exceed window size.")
        self._samples = np.empty(0, dtype=np.float32)
        self._buffer_start_sample = 0
        self._next_window_start = 0
        self.total_samples_received = 0

    @property
    def buffered_samples(self) -> int:
        return len(self._samples)

    @property
    def stream_seconds(self) -> float:
        return self.total_samples_received / self.sample_rate

    def append(self, samples: np.ndarray) -> None:
        values = np.asarray(samples, dtype=np.float32).reshape(-1)
        if values.size:
            self._samples = np.concatenate((self._samples, values))
            self.total_samples_received += len(values)

    def pop_available_windows(self) -> list[AudioWindow]:
        windows = []
        available_end = self._buffer_start_sample + len(self._samples)
        while self._next_window_start + self.window_samples <= available_end:
            relative_start = self._next_window_start - self._buffer_start_sample
            relative_end = relative_start + self.window_samples
            windows.append(AudioWindow(
                samples=self._samples[relative_start:relative_end].copy(),
                start_sample=self._next_window_start,
                end_sample=self._next_window_start + self.window_samples,
                sample_rate=self.sample_rate,
            ))
            self._next_window_start += self.hop_samples

        discard_count = max(0, self._next_window_start - self._buffer_start_sample)
        if discard_count:
            self._samples = self._samples[discard_count:]
            self._buffer_start_sample += discard_count
        return windows
