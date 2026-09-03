"""Deterministic, cautious user-facing fusion summaries."""

from config import FUSION_CONFIG, FusionConfig
from .schemas import Conflict, HypothesisResult


DISPLAY_NAMES = {
    "lubrication_issue": "a possible lubrication-related issue",
    "engine_running_irregularly": "the engine may be running irregularly",
    "cooling_or_combustion_issue": "a possible cooling or combustion-related issue",
    "belt_or_accessory_issue": "a possible belt or accessory-related issue",
    "tire_condition_issue": "a possible tire-condition issue",
    "corrosion_issue": "a possible corrosion issue",
}


def build_summary(
    hypotheses: list[HypothesisResult],
    conflicts: list[Conflict],
    user_description: str | None,
    config: FusionConfig = FUSION_CONFIG,
) -> str:
    strong = [item for item in hypotheses if item.score >= config.strong_hypothesis_score]
    if strong:
        top = strong[0]
        sources = sorted({item.source for item in top.supporting_evidence})
        summary = (
            f"Multiple signals suggest {DISPLAY_NAMES[top.hypothesis]}. "
            f"The finding is supported by {', '.join(sources)} evidence"
        )
        if len(sources) > 1:
            summary += ", which provides cross-source corroboration."
        else:
            summary += "."
    elif hypotheses:
        summary = (
            f"The available evidence weakly supports {DISPLAY_NAMES[hypotheses[0].hypothesis]}, "
            "but it does not strongly support a single issue."
        )
    else:
        summary = (
            "The available evidence is limited or unrelated and does not strongly support a troubleshooting hypothesis."
        )
    if conflicts:
        summary += f" {len(conflicts)} explicit conflict(s) should be resolved with clearer evidence or inspection."
    if user_description and user_description.strip():
        summary += " The user description was retained as context but was not converted into evidence automatically."
    return summary + " These scores are not diagnostic probabilities, and the exact mechanical cause requires inspection."
