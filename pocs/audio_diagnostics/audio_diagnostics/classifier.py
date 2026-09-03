"""Swappable event-classification interface and explainable V1 heuristics."""

from abc import ABC, abstractmethod

from config import HEURISTIC_CONFIG, HeuristicConfig
from .features import ExtractedFeatures, clamp
from .schemas import AudioEvent


class AudioEventClassifier(ABC):
    @abstractmethod
    def predict(self, features: ExtractedFeatures) -> list[AudioEvent]:
        """Interpret measured features as cautious candidate events."""


class HeuristicAudioEventClassifier(AudioEventClassifier):
    def __init__(self, config: HeuristicConfig = HEURISTIC_CONFIG):
        self.config = config

    @staticmethod
    def _score(signals: dict[str, float], weights: dict[str, float]) -> float:
        return clamp(sum(signals[name] * weight for name, weight in weights.items()))

    def _event(self, name: str, score: float, evidence: dict[str, float]) -> AudioEvent:
        return AudioEvent(
            event=name,
            detected=score >= self.config.detection_threshold,
            confidence=score,
            severity=clamp(max(0.0, (score - 0.25) / 0.75)),
            evidence={key: round(value, 4) for key, value in evidence.items()},
        )

    def predict(self, features: ExtractedFeatures) -> list[AudioEvent]:
        s = features.signals
        definitions = [
            ("possible_knocking", self.config.knocking_weights),
            ("possible_hissing", self.config.hissing_weights),
            ("possible_misfire_pattern", self.config.misfire_weights),
            ("possible_bearing_noise", self.config.bearing_weights),
        ]
        return [
            self._event(name, self._score(s, weights), {key: s[key] for key in weights})
            for name, weights in definitions
        ]
