"""Serializable public schemas for video diagnostics."""

from pydantic import BaseModel, Field


class VideoMetadata(BaseModel):
    filename: str
    fps: float
    width: int
    height: int
    frame_count: int
    duration_seconds: float
    frames_analyzed: int
    sample_interval_seconds: float
    timestamps_seconds: list[float]


class VideoEvent(BaseModel):
    event: str
    detected: bool
    confidence: float = Field(ge=0.0, le=1.0)
    severity: float = Field(ge=0.0, le=1.0)
    evidence: dict[str, float]


class VideoEvidence(BaseModel):
    source: str = "video"
    event: str
    severity: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)


class VideoDiagnosticResult(BaseModel):
    status: str = "completed"
    video: VideoMetadata
    events: list[VideoEvent]
    summary: str
    limitations: list[str]

    def to_evidence(self, detected_only: bool = True) -> list[VideoEvidence]:
        events = (event for event in self.events if event.detected or not detected_only)
        return [
            VideoEvidence(event=event.event, severity=event.severity, confidence=event.confidence)
            for event in events
        ]
