"""Conversion of observations into the existing normalized evidence shape."""

from .schemas import ConditionObservation, SymptomObservation, UserReportEvidence


def observations_to_evidence(
    symptoms: list[SymptomObservation], conditions: list[ConditionObservation]
) -> list[UserReportEvidence]:
    condition_names = [item.condition for item in conditions]
    return [
        UserReportEvidence(
            event=item.event,
            severity=item.severity,
            confidence=item.confidence,
            explanation=_explanation(item, condition_names),
            metadata={
                "conditions": condition_names,
                "original_phrase": item.text,
                "uncertain": item.uncertain,
                "negated": False,
            },
        )
        for item in symptoms
        if not item.negated
    ]


def _explanation(item: SymptomObservation, conditions: list[str]) -> str:
    qualifier = "possibly " if item.uncertain else ""
    context = f" under {', '.join(conditions)} conditions" if conditions else ""
    return f"User {qualifier}reported {item.event.replace('_', ' ')}{context}."
