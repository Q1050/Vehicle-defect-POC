from evidence_fusion.service import EvidenceFusionService


def test_smoke_conflict_is_reported_and_penalized():
    base = [{"source": "video", "event": "possible_smoke", "severity": 0.8, "confidence": 0.9}]
    conflict = base + [
        {"source": "user_report", "event": "no_visible_smoke", "severity": 0.8, "confidence": 0.95}
    ]
    service = EvidenceFusionService()
    base_result = service.fuse(base)
    conflict_result = service.fuse(conflict)
    assert len(conflict_result.conflicts) == 1
    assert conflict_result.hypotheses[0].score < base_result.hypotheses[0].score


def test_same_source_is_not_treated_as_independent_conflict():
    result = EvidenceFusionService().fuse([
        {"source": "video", "event": "possible_smoke", "severity": 0.8, "confidence": 0.9},
        {"source": "video", "event": "no_visible_smoke", "severity": 0.8, "confidence": 0.9},
    ])
    assert not result.conflicts
