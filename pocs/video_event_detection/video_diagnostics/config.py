"""Central configuration for local video event heuristics."""

from dataclasses import dataclass


@dataclass(frozen=True)
class VideoConfig:
    supported_extensions: tuple[str, ...] = (".mp4", ".mov", ".avi", ".mkv", ".webm")
    max_duration_seconds: float = 30.0
    min_readable_frames: int = 3
    sample_interval_seconds: float = 0.2
    max_analysis_width: int = 640
    vibration_detection_threshold: float = 0.40
    smoke_detection_threshold: float = 0.52
    motion_reference_fraction: float = 0.012
    residual_motion_fraction: float = 0.004
    motion_exclusion_dilation_pixels: int = 15
    smoke_saturation_max: int = 70
    smoke_value_min: int = 105
    smoke_texture_max: float = 18.0
    smoke_min_area_fraction: float = 0.01
    smoke_max_area_fraction: float = 0.55


VIDEO_CONFIG = VideoConfig()
