"""Pydantic contracts for normalized evidence and fusion results."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class EvidenceItem(BaseModel):
    source: str = Field(min_length=1)
    event: str = Field(min_length=1)
    severity: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    explanation: str | None = None
    origin_id: str | None = None

    @field_validator("source", "event")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = value.strip().lower().replace(" ", "_").replace("-", "_")
        if not normalized:
            raise ValueError("must not be blank")
        return normalized


class GroupedEvidence(BaseModel):
    group_id: str
    source: str
    event: str
    severity: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    source_weight: float = Field(ge=0.0, le=1.0)
    item_count: int = Field(ge=1)
    origin_ids: list[str]
    explanations: list[str]


class Conflict(BaseModel):
    type: str = "conflict"
    description: str
    evidence_ids: list[str]
    events: list[str]


class HypothesisResult(BaseModel):
    hypothesis: str
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    severity: float = Field(ge=0.0, le=1.0)
    supporting_evidence: list[GroupedEvidence]
    conflicting_evidence: list[GroupedEvidence]
    explanation: str


class FusionRequest(BaseModel):
    evidence: list[EvidenceItem] = Field(default_factory=list)
    user_description: str | None = None


class FusionResult(BaseModel):
    status: str = "completed"
    evidence_count: int
    grouped_evidence: list[GroupedEvidence]
    hypotheses: list[HypothesisResult]
    conflicts: list[Conflict]
    summary: str
    limitations: list[str]
