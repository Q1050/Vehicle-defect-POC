import pytest

from evidence_fusion.service import EvidenceFusionService


@pytest.mark.parametrize(
    "evidence, expected",
    [
        ([
            {"source": "audio", "event": "possible_misfire_pattern", "severity": 0.8, "confidence": 0.85},
            {"source": "video", "event": "possible_engine_vibration", "severity": 0.75, "confidence": 0.82},
            {"source": "user_report", "event": "vehicle_shaking", "severity": 0.7, "confidence": 0.9},
        ], "engine_running_irregularly"),
        ([
            {"source": "image", "event": "oil_leak", "severity": 0.8, "confidence": 0.9},
            {"source": "audio", "event": "possible_knocking", "severity": 0.7, "confidence": 0.8},
            {"source": "image", "event": "oil_warning", "severity": 0.8, "confidence": 0.9},
        ], "lubrication_issue"),
        ([
            {"source": "image", "event": "tire_wear", "severity": 0.75, "confidence": 0.88},
            {"source": "image", "event": "low_tire_pressure", "severity": 0.6, "confidence": 0.9},
            {"source": "user_report", "event": "steering_vibration", "severity": 0.6, "confidence": 0.8},
        ], "tire_condition_issue"),
    ],
)
def test_expected_hypothesis_ranks_first(evidence, expected):
    result = EvidenceFusionService().fuse(evidence)
    assert result.hypotheses[0].hypothesis == expected
    assert result.hypotheses[0].score >= 0.55


def test_rust_only_is_present_but_not_overconfident():
    result = EvidenceFusionService().fuse([
        {"source": "image", "event": "rust", "severity": 0.5, "confidence": 0.8}
    ])
    assert result.hypotheses[0].hypothesis == "corrosion_issue"
    assert result.hypotheses[0].score < 0.55


def test_weak_unrelated_evidence_has_no_hypothesis():
    result = EvidenceFusionService().fuse([
        {"source": "user_report", "event": "unrelated_noise", "severity": 0.2, "confidence": 0.3}
    ])
    assert not result.hypotheses
