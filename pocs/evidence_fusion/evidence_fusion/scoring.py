"""Explainable hypothesis score calculations."""

import numpy as np

from config import FUSION_CONFIG, FusionConfig
from .schemas import GroupedEvidence


def clamp(value: float) -> float:
    return round(float(np.clip(value, 0.0, 1.0)), 4)


def noisy_or(values: list[float]) -> float:
    return 1.0 - float(np.prod([1.0 - np.clip(value, 0, 1) for value in values]))


def score_support(
    support: list[GroupedEvidence],
    event_weights: dict[str, float],
    conflict_count: int,
    config: FusionConfig = FUSION_CONFIG,
) -> tuple[float, float, float, float]:
    contributions = [
        group.confidence * group.severity * group.source_weight * event_weights[group.event]
        for group in support
    ]
    confidence_inputs = [
        group.confidence * group.source_weight * event_weights[group.event]
        for group in support
    ]
    independent_sources = len({group.source for group in support})
    corroboration = min(
        config.max_corroboration_bonus,
        max(0, independent_sources - 1) * config.corroboration_bonus_per_source,
    )
    penalty = min(config.conflict_penalty * conflict_count, 0.5)
    base_score = noisy_or(contributions)
    score = min(clamp(base_score + corroboration - penalty), config.maximum_reported_score)
    confidence = min(
        clamp(noisy_or(confidence_inputs) + corroboration - penalty),
        config.maximum_reported_score,
    )
    contribution_total = sum(contributions)
    severity = clamp(
        sum(group.severity * contribution for group, contribution in zip(support, contributions))
        / contribution_total
        if contribution_total
        else 0.0
    )
    return score, confidence, severity, round(corroboration, 4)
