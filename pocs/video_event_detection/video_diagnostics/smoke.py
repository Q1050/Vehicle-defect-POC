"""Lightweight smoke-candidate heuristic using appearance and persistence."""

import cv2
import numpy as np

from .config import VIDEO_CONFIG, VideoConfig
from .motion import clamp
from .schemas import VideoEvent


class SmokeAnalyzer:
    def __init__(self, config: VideoConfig = VIDEO_CONFIG):
        self.config = config

    def _candidate_mask(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        texture = cv2.GaussianBlur(
            np.abs(cv2.Laplacian(gray, cv2.CV_32F)), (9, 9), 0
        )
        mask = (
            (hsv[:, :, 1] <= self.config.smoke_saturation_max)
            & (hsv[:, :, 2] >= self.config.smoke_value_min)
            & (texture <= self.config.smoke_texture_max)
        ).astype(np.uint8) * 255
        kernel = np.ones((7, 7), np.uint8)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    def candidate_masks(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Return masks only for frames with a plausibly sized smoke-like region."""
        masks = [self._candidate_mask(frame) for frame in frames]
        areas = np.array([np.mean(mask > 0) for mask in masks])
        plausible = (
            (areas >= self.config.smoke_min_area_fraction)
            & (areas <= self.config.smoke_max_area_fraction)
        )
        return [mask if is_plausible else np.zeros_like(mask) for mask, is_plausible in zip(masks, plausible)]

    def analyze(
        self,
        frames: list[np.ndarray],
        candidate_masks: list[np.ndarray] | None = None,
    ) -> VideoEvent:
        masks = candidate_masks if candidate_masks is not None else self.candidate_masks(frames)
        areas = np.array([np.mean(mask > 0) for mask in masks])
        plausible = areas > 0
        plausible_area = np.where(plausible, areas, 0.0)
        persistence = float(np.mean(plausible))
        area_score = clamp(float(np.mean(plausible_area)) / 0.18)

        overlaps = []
        changes = []
        upward_values = []
        previous_centroid = None
        for previous, current in zip(masks, masks[1:]):
            intersection = np.count_nonzero((previous > 0) & (current > 0))
            union = np.count_nonzero((previous > 0) | (current > 0))
            overlaps.append(intersection / union if union else 0.0)
            changes.append(float(np.mean(previous != current)))

        for mask in masks:
            moments = cv2.moments(mask)
            centroid = moments["m01"] / moments["m00"] if moments["m00"] else None
            if centroid is not None and previous_centroid is not None:
                upward_values.append(max(0.0, previous_centroid - centroid) / mask.shape[0])
            previous_centroid = centroid

        temporal_persistence = float(np.mean(overlaps)) if overlaps else 0.0
        spreading_motion = clamp((float(np.mean(changes)) if changes else 0.0) / 0.12)
        upward_motion = clamp((float(np.mean(upward_values)) if upward_values else 0.0) / 0.04)
        confidence = clamp(
            0.30 * area_score
            + 0.30 * persistence
            + 0.20 * temporal_persistence
            + 0.15 * spreading_motion
            + 0.05 * upward_motion
        )
        return VideoEvent(
            event="possible_smoke",
            detected=confidence >= self.config.smoke_detection_threshold,
            confidence=confidence,
            severity=clamp(max(0.0, (confidence - 0.20) / 0.80)),
            evidence={
                "candidate_area_fraction": round(float(np.mean(plausible_area)), 4),
                "frame_persistence": round(persistence, 4),
                "temporal_overlap": round(temporal_persistence, 4),
                "spreading_motion_score": round(spreading_motion, 4),
                "upward_motion_score": round(upward_motion, 4),
            },
        )
