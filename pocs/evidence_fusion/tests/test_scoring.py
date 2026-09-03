from config import FUSION_CONFIG
from evidence_fusion.service import EvidenceFusionService


def top_score(evidence):
    return EvidenceFusionService().fuse(evidence).hypotheses[0].score


def test_source_weighting_affects_score():
    image = [{"source": "image", "event": "rust", "severity": 0.7, "confidence": 0.8}]
    user = [{"source": "user_report", "event": "rust", "severity": 0.7, "confidence": 0.8}]
    assert top_score(image) > top_score(user)
    assert FUSION_CONFIG.source_weights["image"] > FUSION_CONFIG.source_weights["user_report"]


def test_independent_source_corroboration_increases_score():
    one = [{"source": "audio", "event": "possible_misfire_pattern", "severity": 0.7, "confidence": 0.8}]
    multiple = one + [
        {"source": "video", "event": "possible_engine_vibration", "severity": 0.7, "confidence": 0.8}
    ]
    assert top_score(multiple) > top_score(one)


def test_scores_remain_clamped():
    evidence = [
        {"source": source, "event": "possible_misfire_pattern", "severity": 1, "confidence": 1}
        for source in ("image", "audio", "video", "telemetry")
    ]
    result = EvidenceFusionService().fuse(evidence)
    assert all(0 <= item.score <= 1 for item in result.hypotheses)
