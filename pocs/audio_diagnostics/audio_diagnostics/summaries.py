"""Deterministic, non-LLM diagnostic summaries."""

from .schemas import AudioEvent, OverallAssessment


EVENT_PHRASES = {
    "possible_knocking": "repeated transient sounds that may be consistent with knocking or another intermittent mechanical sound",
    "possible_hissing": "sustained high-frequency, noise-like energy that may be consistent with hissing",
    "possible_misfire_pattern": "irregular pulse timing that may be consistent with a misfire-like pattern",
    "possible_bearing_noise": "persistent tonal or repetitive mechanical energy that may be consistent with bearing-related noise",
}


def build_summary(
    overall: OverallAssessment,
    events: list[AudioEvent],
    user_description: str | None = None,
) -> str:
    detected = sorted((event for event in events if event.detected), key=lambda item: item.confidence, reverse=True)
    if detected:
        findings = "; ".join(EVENT_PHRASES[event.event] for event in detected[:2])
        summary = f"The recording contains {findings}."
    else:
        summary = "No strong abnormal audio pattern was identified in this recording."

    if user_description and user_description.strip():
        summary += " The supplied description was retained as context but did not change the acoustic measurements."
    if overall.anomaly_detected:
        summary += " This POC cannot determine the mechanical cause from audio alone, so confirm the result through inspection."
    else:
        summary += " If symptoms persist, try a clearer recording close to the engine and seek a mechanical inspection."
    return summary
