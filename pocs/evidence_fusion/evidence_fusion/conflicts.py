"""Limited, explicit contradiction detection."""

from config import FUSION_CONFIG, FusionConfig
from .rules import CONFLICT_RULES
from .schemas import Conflict, GroupedEvidence


def detect_conflicts(
    groups: list[GroupedEvidence],
    config: FusionConfig = FUSION_CONFIG,
) -> tuple[list[Conflict], dict[str, list[GroupedEvidence]]]:
    strong = [
        group for group in groups
        if group.confidence * group.source_weight >= config.conflict_strength_threshold
    ]
    conflicts = []
    conflicting_by_event: dict[str, list[GroupedEvidence]] = {}
    for positive_events, negative_events, description in CONFLICT_RULES:
        positives = [group for group in strong if group.event in positive_events]
        negatives = [group for group in strong if group.event in negative_events]
        for positive in positives:
            for negative in negatives:
                if positive.source == negative.source:
                    continue
                conflicts.append(Conflict(
                    description=description,
                    evidence_ids=[positive.group_id, negative.group_id],
                    events=[positive.event, negative.event],
                ))
                conflicting_by_event.setdefault(positive.event, []).append(negative)
                conflicting_by_event.setdefault(negative.event, []).append(positive)
    return conflicts, conflicting_by_event
