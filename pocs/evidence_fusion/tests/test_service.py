from evidence_fusion.schemas import FusionResult
from evidence_fusion.service import EvidenceFusionService


def test_empty_evidence_returns_completed_weak_summary():
    result = EvidenceFusionService().fuse([])
    assert result.status == "completed"
    assert result.evidence_count == 0
    assert not result.hypotheses
    assert "limited" in result.summary


def test_service_serialization_and_deterministic_summary():
    evidence = [
        {"source": "audio", "event": "possible_misfire_pattern", "severity": 0.8, "confidence": 0.85},
        {"source": "video", "event": "possible_engine_vibration", "severity": 0.75, "confidence": 0.82},
    ]
    service = EvidenceFusionService()
    first = service.fuse(evidence, "The car shakes.")
    second = service.fuse(evidence, "The car shakes.")
    payload = first.model_dump()
    assert first.summary == second.summary
    assert isinstance(FusionResult.model_validate(payload), FusionResult)
    assert "not converted into evidence" in first.summary
