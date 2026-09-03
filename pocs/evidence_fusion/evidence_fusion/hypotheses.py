"""Rule-backed hypothesis ranking."""

from config import FUSION_CONFIG, FusionConfig
from .rules import HYPOTHESIS_RULES
from .scoring import score_support
from .schemas import GroupedEvidence, HypothesisResult


def build_hypotheses(
    groups: list[GroupedEvidence],
    conflicting_by_event: dict[str, list[GroupedEvidence]],
    config: FusionConfig = FUSION_CONFIG,
) -> list[HypothesisResult]:
    results = []
    for hypothesis, event_weights in HYPOTHESIS_RULES.items():
        support = [group for group in groups if group.event in event_weights]
        if not support:
            continue
        conflicting = []
        for group in support:
            conflicting.extend(conflicting_by_event.get(group.event, []))
        conflicting = list({group.group_id: group for group in conflicting}.values())
        score, confidence, severity, bonus = score_support(
            support, event_weights, len(conflicting), config
        )
        if score < config.minimum_hypothesis_score:
            continue
        source_count = len({group.source for group in support})
        explanation = (
            f"{len(support)} grouped signal(s) from {source_count} independent source(s) "
            f"support this hypothesis; corroboration bonus={bonus:.2f}."
        )
        if conflicting:
            explanation += f" {len(conflicting)} conflicting signal(s) reduced the score."
        results.append(HypothesisResult(
            hypothesis=hypothesis,
            score=score,
            confidence=confidence,
            severity=severity,
            supporting_evidence=sorted(support, key=lambda group: group.confidence, reverse=True),
            conflicting_evidence=conflicting,
            explanation=explanation,
        ))
    return sorted(results, key=lambda result: result.score, reverse=True)
