"""PCM conversion and in-memory WAV encoding."""

from io import BytesIO
import wave

import numpy as np


def pcm_s16le_to_float32(payload: bytes) -> np.ndarray:
    if not payload:
        raise ValueError("Audio chunk is empty.")
    if len(payload) % 2:
        raise ValueError("PCM16 chunks must contain an even number of bytes.")
    return (np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0).copy()


def float32_to_pcm_s16le(samples: np.ndarray) -> bytes:
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    return np.round(clipped * 32767.0).astype("<i2").tobytes()


def samples_to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(float32_to_pcm_s16le(samples))
    return buffer.getvalue()
