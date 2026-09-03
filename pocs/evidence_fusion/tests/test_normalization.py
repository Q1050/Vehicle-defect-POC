import pytest
from pydantic import ValidationError

from evidence_fusion.normalization import (
    canonicalize_event,
    group_evidence,
    image_detections_to_evidence,
    normalize_evidence,
)
from evidence_fusion.schemas import EvidenceItem


def test_schema_validates_score_ranges():
    with pytest.raises(ValidationError):
        EvidenceItem(source="image", event="rust", severity=1.2, confidence=0.8)


def test_alias_and_image_adapter():
    assert canonicalize_event("oil warning") == "low_engine_oil_warning"
    adapted = image_detections_to_evidence([
        {"label": "oil_leak", "confidence": 0.81, "bbox": [1, 2, 3, 4]}
    ])
    assert adapted[0].source == "image"
    assert adapted[0].severity == 0.5
    assert adapted[0].metadata["bbox"] == [1, 2, 3, 4]


def test_same_source_duplicates_are_grouped_with_small_increment():
    normalized = normalize_evidence([
        {"source": "image", "event": "rust", "severity": 0.6, "confidence": 0.8},
        {"source": "image", "event": "rust", "severity": 0.7, "confidence": 0.7},
        {"source": "image", "event": "rust", "severity": 0.5, "confidence": 0.6},
    ])
    groups = group_evidence(normalized)
    assert len(groups) == 1
    assert groups[0].item_count == 3
    assert groups[0].confidence == pytest.approx(0.995)
    assert groups[0].confidence < 0.8 + 0.7 + 0.6
