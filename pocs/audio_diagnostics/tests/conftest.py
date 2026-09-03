from io import BytesIO

import numpy as np
import pytest
import soundfile as sf


SAMPLE_RATE = 22_050


def wav_bytes(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    buffer = BytesIO()
    sf.write(buffer, samples, sample_rate, format="WAV")
    return buffer.getvalue()


@pytest.fixture
def clean_tone() -> np.ndarray:
    t = np.arange(SAMPLE_RATE * 3) / SAMPLE_RATE
    return (0.35 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)


@pytest.fixture
def repeated_impulses() -> np.ndarray:
    signal = np.zeros(SAMPLE_RATE * 4, dtype=np.float32)
    for position in np.arange(0.2, 3.9, 0.25):
        start = int(position * SAMPLE_RATE)
        signal[start:start + 150] = np.hanning(150)
    return signal


@pytest.fixture
def high_frequency_noise() -> np.ndarray:
    rng = np.random.default_rng(11)
    noise = rng.normal(0, 0.3, SAMPLE_RATE * 4)
    spectrum = np.fft.rfft(noise)
    frequencies = np.fft.rfftfreq(len(noise), 1 / SAMPLE_RATE)
    spectrum[frequencies < 3_000] = 0
    return np.fft.irfft(spectrum).astype(np.float32)
