"""Public, serializable result schemas."""

from pydantic import BaseModel, Field


class AudioMetadata(BaseModel):
    filename: str
    duration_seconds: float
    sample_rate: int
    samples_analyzed: int


class OverallAssessment(BaseModel):
    anomaly_detected: bool
    anomaly_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)


class AudioEvent(BaseModel):
    event: str
    detected: bool
    confidence: float = Field(ge=0.0, le=1.0)
    severity: float = Field(ge=0.0, le=1.0)
    evidence: dict[str, float]


class FeatureSummary(BaseModel):
    rms: dict[str, float]
    zero_crossing_rate: dict[str, float]
    spectral_centroid: dict[str, float]
    spectral_bandwidth: dict[str, float]
    spectral_rolloff: dict[str, float]
    mfcc_summary: list[dict[str, float]]
    onset_strength: dict[str, float]
    temporal_energy_variation: float
    frequency_band_energy: dict[str, float]
    measured_signals: dict[str, float]


class EvidenceItem(BaseModel):
    source: str = "audio"
    event: str
    severity: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)


class AudioDiagnosticResult(BaseModel):
    status: str = "completed"
    audio: AudioMetadata
    overall: OverallAssessment
    events: list[AudioEvent]
    features: FeatureSummary
    transcription: str | None = None
    summary: str
    limitations: list[str]

    def to_evidence(self, detected_only: bool = True) -> list[EvidenceItem]:
        events = (event for event in self.events if event.detected or not detected_only)
        return [
            EvidenceItem(event=event.event, severity=event.severity, confidence=event.confidence)
            for event in events
        ]
