"""OpenCV video validation, metadata extraction, and frame sampling."""

from dataclasses import dataclass
from pathlib import Path
import os
import tempfile

import cv2
import numpy as np

from .config import VIDEO_CONFIG, VideoConfig
from .schemas import VideoMetadata


class VideoValidationError(ValueError):
    """Raised when video input cannot be analyzed safely."""


@dataclass(frozen=True)
class SampledVideo:
    metadata: VideoMetadata
    frames: list[np.ndarray]


class VideoFrameSampler:
    def __init__(self, config: VideoConfig = VIDEO_CONFIG):
        self.config = config

    def sample(self, video_bytes: bytes, filename: str) -> SampledVideo:
        extension = Path(filename).suffix.lower()
        if extension not in self.config.supported_extensions:
            raise VideoValidationError(
                f"Unsupported video format '{extension or 'unknown'}'. Supported: "
                f"{', '.join(self.config.supported_extensions)}."
            )
        if not video_bytes:
            raise VideoValidationError("The uploaded video is empty.")

        temp_path = None
        capture = None
        try:
            with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as temp_file:
                temp_path = temp_file.name
                temp_file.write(video_bytes)

            capture = cv2.VideoCapture(temp_path)
            if not capture.isOpened():
                raise VideoValidationError("The uploaded video could not be opened by OpenCV.")

            fps = float(capture.get(cv2.CAP_PROP_FPS))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps <= 0 or width <= 0 or height <= 0:
                raise VideoValidationError("The video has invalid or unavailable stream metadata.")
            duration = frame_count / fps if frame_count > 0 else 0.0
            if duration > self.config.max_duration_seconds:
                raise VideoValidationError(
                    f"Video is {duration:.2f}s; maximum is {self.config.max_duration_seconds:.0f}s."
                )

            frame_step = max(1, round(fps * self.config.sample_interval_seconds))
            frames: list[np.ndarray] = []
            timestamps: list[float] = []
            index = 0
            while True:
                readable, frame = capture.read()
                if not readable:
                    break
                if index % frame_step == 0:
                    if frame.shape[1] > self.config.max_analysis_width:
                        scale = self.config.max_analysis_width / frame.shape[1]
                        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    frames.append(frame)
                    timestamps.append(index / fps)
                index += 1

            if len(frames) < self.config.min_readable_frames:
                raise VideoValidationError(
                    f"Only {len(frames)} sampled frames were readable; "
                    f"at least {self.config.min_readable_frames} are required."
                )
            actual_count = frame_count if frame_count > 0 else index
            actual_duration = actual_count / fps
            metadata = VideoMetadata(
                filename=filename,
                fps=round(fps, 3),
                width=width,
                height=height,
                frame_count=actual_count,
                duration_seconds=round(actual_duration, 3),
                frames_analyzed=len(frames),
                sample_interval_seconds=round(frame_step / fps, 4),
                timestamps_seconds=[round(value, 3) for value in timestamps],
            )
            return SampledVideo(metadata=metadata, frames=frames)
        finally:
            if capture is not None:
                capture.release()
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
