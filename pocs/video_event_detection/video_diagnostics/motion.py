"""Explainable residual optical-flow vibration heuristic."""

import cv2
import numpy as np

from .config import VIDEO_CONFIG, VideoConfig
from .schemas import VideoEvent


def clamp(value: float) -> float:
    return round(float(np.clip(value, 0.0, 1.0)), 4)


class VibrationAnalyzer:
    def __init__(self, config: VideoConfig = VIDEO_CONFIG):
        self.config = config

    def analyze(
        self,
        frames: list[np.ndarray],
        exclusion_masks: list[np.ndarray] | None = None,
    ) -> VideoEvent:
        residual_series = []
        global_series = []
        localized_series = []

        previous = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        diagonal = float(np.hypot(previous.shape[0], previous.shape[1]))
        for index, frame in enumerate(frames[1:], start=1):
            current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                previous, current, None, 0.5, 3, 21, 3, 5, 1.2, 0
            )
            previous_exclusion = None
            current_exclusion = None
            if exclusion_masks is not None:
                if len(exclusion_masks) != len(frames):
                    raise ValueError("exclusion_masks must contain one mask per frame")
                kernel_size = self.config.motion_exclusion_dilation_pixels
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                previous_exclusion = cv2.dilate(exclusion_masks[index - 1], kernel)
                current_exclusion = cv2.dilate(exclusion_masks[index], kernel)
            feature_mask = None if previous_exclusion is None else cv2.bitwise_not(previous_exclusion)
            points = cv2.goodFeaturesToTrack(
                previous,
                maxCorners=250,
                qualityLevel=0.01,
                minDistance=7,
                mask=feature_mask,
            )
            global_vector = np.zeros(2, dtype=np.float32)
            if points is not None and len(points) >= 4:
                tracked, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None)
                valid = status.reshape(-1).astype(bool) if status is not None else np.zeros(len(points), dtype=bool)
                if tracked is not None and np.count_nonzero(valid) >= 4:
                    displacements = tracked[valid].reshape(-1, 2) - points[valid].reshape(-1, 2)
                    global_vector = np.median(displacements, axis=0)
            global_magnitude = float(np.linalg.norm(global_vector)) / diagonal
            residual = flow - global_vector
            residual_magnitude = np.linalg.norm(residual, axis=2) / diagonal
            texture_mask = cv2.magnitude(
                cv2.Sobel(previous, cv2.CV_32F, 1, 0),
                cv2.Sobel(previous, cv2.CV_32F, 0, 1),
            ) > 10
            if previous_exclusion is not None:
                excluded = (previous_exclusion > 0) | (current_exclusion > 0)
                texture_mask &= ~excluded
            measured_residual = residual_magnitude[texture_mask] if np.any(texture_mask) else residual_magnitude.ravel()
            residual_mean = float(np.mean(measured_residual))
            active_fraction = float(np.mean(measured_residual > self.config.residual_motion_fraction))
            global_series.append(global_magnitude)
            residual_series.append(residual_mean)
            localized_series.append(active_fraction)
            previous = current

        residual_array = np.asarray(residual_series)
        mean_residual = float(np.mean(residual_array))
        motion_variance = float(np.std(residual_array))
        mean_global = float(np.mean(global_series))
        localized_fraction = float(np.mean(localized_series))
        if len(residual_array) >= 4 and np.std(residual_array) > 1e-8:
            centered = residual_array - np.mean(residual_array)
            autocorrelation = np.correlate(centered, centered, mode="full")[len(centered) - 1:]
            autocorrelation /= autocorrelation[0] + 1e-12
            repetitive = float(np.max(autocorrelation[1:min(len(autocorrelation), 8)], initial=0.0))
        else:
            repetitive = 0.0

        residual_score = clamp(mean_residual / self.config.motion_reference_fraction)
        variance_score = clamp(motion_variance / (self.config.motion_reference_fraction * 0.6))
        localized_score = clamp(localized_fraction / 0.30)
        repetitive_score = clamp(max(repetitive, 0.0))
        camera_dominance = clamp(mean_global / (mean_global + mean_residual + 1e-9))
        confidence = clamp(
            (0.45 * residual_score + 0.25 * variance_score + 0.20 * localized_score
             + 0.10 * repetitive_score)
            * (1.0 - 0.45 * camera_dominance)
        )
        return VideoEvent(
            event="possible_engine_vibration",
            detected=confidence >= self.config.vibration_detection_threshold,
            confidence=confidence,
            severity=clamp(max(0.0, (confidence - 0.20) / 0.80)),
            evidence={
                "mean_residual_motion": round(mean_residual, 6),
                "motion_variance": round(motion_variance, 6),
                "repetitive_motion_score": round(repetitive_score, 4),
                "localized_motion_fraction": round(localized_fraction, 4),
                "global_camera_motion": round(mean_global, 6),
                "camera_motion_dominance": round(camera_dominance, 4),
            },
        )
