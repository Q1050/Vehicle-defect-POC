"""Deterministic, non-diagnostic consumer summaries."""

from .schemas import ConditionObservation, SymptomObservation, TemporalInformation


def build_summary(
    symptoms: list[SymptomObservation], conditions: list[ConditionObservation], temporal: TemporalInformation
) -> str:
    positive = [item.event.replace("_", " ") for item in symptoms if not item.negated]
    negative = [item.event.replace("_", " ") for item in symptoms if item.negated]
    parts: list[str] = []
    if positive:
        names = _join(positive)
        context = f" during {_join([item.condition.replace('_', ' ') for item in conditions])}" if conditions else ""
        parts.append(f"The driver reported {names}{context}.")
    if negative:
        parts.append(f"The driver explicitly did not report {_join(negative)}.")
    if not positive and not negative:
        parts.append("No supported vehicle symptoms were explicitly identified in the description.")
    if temporal.trend == "improves_when_warm":
        parts.append("The symptoms reportedly improve after the vehicle warms up.")
    elif temporal.trend:
        parts.append(f"The symptoms were described as {temporal.trend.replace('_', ' ')}.")
    parts.append("These are user-reported observations and have not been mechanically verified.")
    return " ".join(parts)


def _join(items: list[str]) -> str:
    if len(items) < 2:
        return items[0] if items else ""
    return ", ".join(items[:-1]) + f" and {items[-1]}"
