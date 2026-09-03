"""Deterministic summaries that do not require an external model."""

from .schemas import VideoEvent


def build_summary(events: list[VideoEvent], user_description: str | None = None) -> str:
    detected = {event.event for event in events if event.detected}
    parts = []
    if "possible_engine_vibration" in detected:
        parts.append(
            "Repeated residual motion was observed after accounting for broad frame movement. "
            "This may indicate unstable vibration, although handheld camera motion can produce similar patterns."
        )
    if "possible_smoke" in detected:
        parts.append(
            "A persistent low-saturation, low-texture moving region was observed across sampled frames. "
            "This may be consistent with visible smoke, but lighting, steam, and reflections can appear similar."
        )
    if not parts:
        parts.append(
            "No strong unstable motion or persistent smoke-like pattern was identified in the analyzed frames."
        )
    if user_description and user_description.strip():
        parts.append(
            "The supplied description was retained as context and did not alter the measured video signals."
        )
    parts.append("These POC findings are not a mechanical diagnosis and should be confirmed by inspection.")
    return " ".join(parts)
