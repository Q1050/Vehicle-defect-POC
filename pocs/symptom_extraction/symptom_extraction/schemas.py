"""Stable Pydantic contracts for extraction and Evidence Fusion compatibility."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class SymptomObservation(BaseModel):
    event: str
    text: str
    confidence: float = Field(ge=0, le=1)
    severity: float = Field(ge=0, le=1)
    negated: bool = False
    uncertain: bool = False


class ConditionObservation(BaseModel):
    condition: str
    text: str
    confidence: float = Field(ge=0, le=1)


class TemporalInformation(BaseModel):
    onset: str | None = None
    duration: str | None = None
    frequency: str | None = None
    trend: str | None = None


class UserReportEvidence(BaseModel):
    source: Literal["user_report"] = "user_report"
    event: str
    severity: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    explanation: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SymptomExtractionResult(BaseModel):
    status: Literal["completed"] = "completed"
    original_text: str
    symptoms: list[SymptomObservation] = Field(default_factory=list)
    conditions: list[ConditionObservation] = Field(default_factory=list)
    temporal: TemporalInformation = Field(default_factory=TemporalInformation)
    evidence: list[UserReportEvidence] = Field(default_factory=list)
    summary: str
    limitations: list[str] = Field(default_factory=list)

    def to_evidence(self, include_negated: bool = False) -> list[UserReportEvidence]:
        if not include_negated:
            return list(self.evidence)
        existing = {item.event for item in self.evidence}
        result = list(self.evidence)
        for symptom in self.symptoms:
            if symptom.negated and f"no_{symptom.event}" not in existing:
                result.append(UserReportEvidence(
                    event=f"no_{symptom.event}", severity=symptom.severity,
                    confidence=symptom.confidence,
                    explanation=f'User explicitly denied "{symptom.text}".',
                    metadata={"negated": True, "original_phrase": symptom.text},
                ))
        return result
