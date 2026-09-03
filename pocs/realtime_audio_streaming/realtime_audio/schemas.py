"""Serializable WebSocket protocol and live-result schemas."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class StartMessage(BaseModel):
    type: Literal["start"]
    sample_rate: int
    channels: int
    sample_format: str
    user_description: str | None = None


class StopMessage(BaseModel):
    type: Literal["stop"]


class WindowRange(BaseModel):
    start_seconds: float
    end_seconds: float


class LiveEvent(BaseModel):
    event: str
    detected: bool
    confidence: float = Field(ge=0.0, le=1.0)
    severity: float = Field(ge=0.0, le=1.0)
    evidence: dict[str, float] = Field(default_factory=dict)


class StabilizedEvent(BaseModel):
    event: str
    active: bool
    state: Literal["inactive", "started", "ongoing", "ended"]
    confidence: float = Field(ge=0.0, le=1.0)
    severity: float = Field(ge=0.0, le=1.0)


class EvidenceItem(BaseModel):
    source: Literal["audio"] = "audio"
    event: str
    severity: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)


class DiagnosticUpdate(BaseModel):
    type: Literal["diagnostic_update"] = "diagnostic_update"
    session_id: str
    window: WindowRange
    overall: dict[str, Any]
    events: list[LiveEvent]
    stabilized_events: list[StabilizedEvent]
    evidence: list[EvidenceItem]
    summary: str


class SessionStarted(BaseModel):
    type: Literal["session_started"] = "session_started"
    session_id: str


class SessionEnded(BaseModel):
    type: Literal["session_ended"] = "session_ended"
    session_id: str
    duration_seconds: float
    windows_processed: int


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    code: str
    message: str
