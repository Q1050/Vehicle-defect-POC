"""Replaceable overall anomaly-scoring engine."""

from abc import ABC, abstractmethod

import numpy as np

from config import HEURISTIC_CONFIG, HeuristicConfig
from .features import ExtractedFeatures, clamp
from .schemas import AudioEvent, OverallAssessment


class AnomalyEngine(ABC):
    @abstractmethod
    def score(self, features: ExtractedFeatures, events: list[AudioEvent]) -> OverallAssessment:
        """Return a normalized overall anomaly assessment."""


class HeuristicAnomalyEngine(AnomalyEngine):
    def __init__(self, config: HeuristicConfig = HEURISTIC_CONFIG):
        self.config = config

    def score(self, features: ExtractedFeatures, events: list[AudioEvent]) -> OverallAssessment:
        event_component = max((event.confidence for event in events), default=0.0)
        general_keys = (
            "transient_score", "high_frequency_energy", "temporal_energy_variation",
            "energy_persistence",
        )
        signal_component = float(np.mean([features.signals[key] for key in general_keys]))
        anomaly_score = clamp(
            event_component * self.config.overall_event_weight
            + signal_component * self.config.overall_signal_weight
        )
        confidence = clamp(0.55 * event_component + 0.45 * abs(anomaly_score - 0.5) * 2.0)
        return OverallAssessment(
            anomaly_detected=any(event.detected for event in events),
            anomaly_score=anomaly_score,
            confidence=confidence,
        )
