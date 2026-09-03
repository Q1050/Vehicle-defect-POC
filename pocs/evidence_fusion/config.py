"""Central POC assumptions for evidence fusion."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FusionConfig:
    source_weights: dict[str, float] = field(default_factory=lambda: {
        "image": 1.00,
        "audio": 0.90,
        "video": 0.90,
        "user_report": 0.65,
        "manual": 0.75,
        "telemetry": 1.00,
    })
    unknown_source_weight: float = 0.60
    duplicate_additional_weight: float = 0.15
    corroboration_bonus_per_source: float = 0.08
    max_corroboration_bonus: float = 0.16
    conflict_penalty: float = 0.18
    conflict_strength_threshold: float = 0.55
    minimum_hypothesis_score: float = 0.12
    strong_hypothesis_score: float = 0.55
    maximum_reported_score: float = 0.95
    image_default_severity: float = 0.50


FUSION_CONFIG = FusionConfig()
