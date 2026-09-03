"""Audio decoding and signal preprocessing independent of any web framework."""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from config import AUDIO_CONFIG, AudioConfig


class AudioValidationError(ValueError):
    """Raised when uploaded audio cannot be safely analyzed."""


@dataclass(frozen=True)
class ProcessedAudio:
    samples: np.ndarray
    sample_rate: int
    duration_seconds: float


class AudioPreprocessor:
    def __init__(self, config: AudioConfig = AUDIO_CONFIG):
        self.config = config

    def process(self, audio_bytes: bytes, filename: str) -> ProcessedAudio:
        extension = Path(filename).suffix.lower()
        if extension not in self.config.supported_extensions:
            raise AudioValidationError(
                f"Unsupported audio format '{extension or 'unknown'}'. Supported: "
                f"{', '.join(self.config.supported_extensions)}."
            )
        if not audio_bytes:
            raise AudioValidationError("The uploaded audio file is empty.")

        try:
            samples, source_rate = sf.read(BytesIO(audio_bytes), dtype="float32", always_2d=True)
        except Exception as exc:
            raise AudioValidationError(
                "The audio could not be decoded. MP3/M4A support may require an FFmpeg-enabled audio backend."
            ) from exc

        if samples.size == 0 or source_rate <= 0:
            raise AudioValidationError("The uploaded file contains no readable audio samples.")
        if not np.isfinite(samples).all():
            raise AudioValidationError("The audio contains invalid numeric samples.")

        mono = np.mean(samples, axis=1, dtype=np.float32)
        original_duration = len(mono) / source_rate
        if original_duration > self.config.max_duration_seconds:
            raise AudioValidationError(
                f"Audio is {original_duration:.2f}s; maximum is {self.config.max_duration_seconds:.0f}s."
            )
        if original_duration < self.config.min_duration_seconds:
            raise AudioValidationError(
                f"Audio is {original_duration:.2f}s; minimum is {self.config.min_duration_seconds:.1f}s."
            )

        if source_rate != self.config.target_sample_rate:
            mono = librosa.resample(
                mono, orig_sr=source_rate, target_sr=self.config.target_sample_rate
            ).astype(np.float32)

        trimmed, _ = librosa.effects.trim(mono, top_db=self.config.trim_top_db)
        if trimmed.size == 0 or float(np.max(np.abs(trimmed), initial=0.0)) < 1e-8:
            raise AudioValidationError("The recording is silent or contains no analyzable signal.")

        peak = float(np.max(np.abs(trimmed)))
        normalized = (trimmed * (self.config.normalization_peak / peak)).astype(np.float32)
        duration = len(normalized) / self.config.target_sample_rate
        if duration < self.config.min_duration_seconds:
            raise AudioValidationError("Too little non-silent audio remains after trimming.")

        return ProcessedAudio(normalized, self.config.target_sample_rate, duration)
