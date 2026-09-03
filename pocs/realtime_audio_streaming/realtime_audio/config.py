"""Central configuration for the realtime audio streaming package."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RealtimeAudioConfig:
    sample_rate: int = 16_000
    channels: int = 1
    sample_format: str = "pcm_s16le"
    sample_width_bytes: int = 2
    window_seconds: float = 4.0
    hop_seconds: float = 1.0
    stabilization_windows: int = 3
    activation_consecutive_windows: int = 2
    deactivation_consecutive_windows: int = 2
    max_session_seconds: float = 120.0
    max_chunk_bytes: int = 256_000


REALTIME_CONFIG = RealtimeAudioConfig()
