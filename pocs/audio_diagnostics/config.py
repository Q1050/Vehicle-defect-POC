"""Central configuration for the audio diagnostics POC."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AudioConfig:
    target_sample_rate: int = 22_050
    min_duration_seconds: float = 0.5
    max_duration_seconds: float = 30.0
    trim_top_db: float = 35.0
    normalization_peak: float = 0.95
    n_fft: int = 2_048
    hop_length: int = 512
    n_mfcc: int = 13
    low_band_hz: tuple[float, float] = (40.0, 400.0)
    mid_band_hz: tuple[float, float] = (400.0, 2_500.0)
    high_band_hz: tuple[float, float] = (2_500.0, 10_000.0)
    supported_extensions: tuple[str, ...] = (".wav", ".mp3", ".m4a", ".flac")


@dataclass(frozen=True)
class HeuristicConfig:
    detection_threshold: float = 0.58
    knocking_weights: dict[str, float] = field(default_factory=lambda: {
        "transient_score": 0.40, "low_mid_energy": 0.35, "impulse_rate_score": 0.25,
    })
    hissing_weights: dict[str, float] = field(default_factory=lambda: {
        "high_frequency_energy": 0.45, "spectral_flatness": 0.30, "high_frequency_persistence": 0.25,
    })
    misfire_weights: dict[str, float] = field(default_factory=lambda: {
        "onset_interval_irregularity": 0.45, "temporal_energy_variation": 0.35, "pulse_activity": 0.20,
    })
    bearing_weights: dict[str, float] = field(default_factory=lambda: {
        "tonal_concentration": 0.35, "mid_frequency_energy": 0.25, "energy_persistence": 0.25,
        "repetitive_component": 0.15,
    })
    overall_event_weight: float = 0.75
    overall_signal_weight: float = 0.25


AUDIO_CONFIG = AudioConfig()
HEURISTIC_CONFIG = HeuristicConfig()
