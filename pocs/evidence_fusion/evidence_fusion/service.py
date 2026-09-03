"""Framework-independent evidence-fusion orchestration."""

from typing import Any

from .conflicts import detect_conflicts
from .hypotheses import build_hypotheses
from .normalization import group_evidence, normalize_evidence
from .schemas import EvidenceItem, FusionResult
from .summaries import build_summary


LIMITATIONS = [
    "Source weights, event mappings, bonuses, and penalties are POC assumptions, not statistically calibrated values.",
    "Fusion improves organization and corroboration; it does not establish a mechanical diagnosis.",
    "Incorrect upstream detections can reinforce one another and produce a misleading high score.",
    "Manual/RAG context can be added later as structured evidence but is not retrieved by this service.",
]


class EvidenceFusionService:
    def fuse(
        self,
        evidence: list[EvidenceItem | dict[str, Any]],
        user_description: str | None = None,
    ) -> FusionResult:
        normalized = normalize_evidence(evidence)
        grouped = group_evidence(normalized)
        conflicts, conflicting_by_event = detect_conflicts(grouped)
        hypotheses = build_hypotheses(grouped, conflicting_by_event)
        return FusionResult(
            evidence_count=len(normalized),
            grouped_evidence=grouped,
            hypotheses=hypotheses,
            conflicts=conflicts,
            summary=build_summary(hypotheses, conflicts, user_description),
            limitations=LIMITATIONS,
        )
