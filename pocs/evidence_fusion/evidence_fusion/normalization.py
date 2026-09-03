"""Explicit event canonicalization, image adaptation, and within-source grouping."""

from collections import defaultdict
from typing import Any

import numpy as np

from config import FUSION_CONFIG, FusionConfig
from .schemas import EvidenceItem, GroupedEvidence


EVENT_ALIASES = {
    "engine_warning": "check_engine_warning",
    "check_engine": "check_engine_warning",
    "check_engine_light": "check_engine_warning",
    "low_engine_oil": "low_engine_oil_warning",
    "oil_warning": "low_engine_oil_warning",
    "oil_pressure_warning": "low_engine_oil_warning",
    "overheating": "overheating_warning",
    "temperature_warning": "overheating_warning",
    "low_tire_pressure": "tire_pressure_warning",
    "tpms_warning": "tire_pressure_warning",
    "car_shaking": "vehicle_shaking",
    "engine_shaking": "vehicle_shaking",
    "steering_vibration": "steering_vibration",
    "visible_corrosion": "rust",
    "no_smoke": "no_visible_smoke",
    "no_dashboard_warnings": "no_warning_lights",
}


def canonicalize_event(event: str) -> str:
    normalized = event.strip().lower().replace(" ", "_").replace("-", "_")
    return EVENT_ALIASES.get(normalized, normalized)


def normalize_evidence(items: list[EvidenceItem | dict[str, Any]]) -> list[EvidenceItem]:
    normalized = []
    for item in items:
        evidence = item if isinstance(item, EvidenceItem) else EvidenceItem.model_validate(item)
        normalized.append(evidence.model_copy(update={"event": canonicalize_event(evidence.event)}))
    return normalized


def image_detections_to_evidence(
    detections: list[dict[str, Any]],
    default_severity: float = FUSION_CONFIG.image_default_severity,
    origin_id: str | None = None,
) -> list[EvidenceItem]:
    """Adapt current YOLO label/confidence dictionaries without changing that POC."""
    return [
        EvidenceItem(
            source="image",
            event=canonicalize_event(str(detection["label"])),
            confidence=float(detection["confidence"]),
            severity=float(detection.get("severity", default_severity)),
            origin_id=origin_id,
            metadata={"bbox": detection["bbox"]} if "bbox" in detection else {},
        )
        for detection in detections
    ]


def group_evidence(
    items: list[EvidenceItem],
    config: FusionConfig = FUSION_CONFIG,
) -> list[GroupedEvidence]:
    buckets: dict[tuple[str, str], list[tuple[int, EvidenceItem]]] = defaultdict(list)
    for index, item in enumerate(items):
        buckets[(item.source, item.event)].append((index, item))

    groups = []
    for (source, event), indexed_items in buckets.items():
        ordered = sorted(indexed_items, key=lambda pair: pair[1].confidence, reverse=True)
        primary = ordered[0][1]
        extra_confidence = sum(item.confidence for _, item in ordered[1:])
        combined_confidence = min(
            1.0,
            primary.confidence + config.duplicate_additional_weight * extra_confidence,
        )
        confidence_total = sum(item.confidence for _, item in ordered)
        severity = (
            sum(item.severity * item.confidence for _, item in ordered) / confidence_total
            if confidence_total
            else float(np.mean([item.severity for _, item in ordered]))
        )
        ids = [item.origin_id or f"evidence-{index + 1}" for index, item in ordered]
        groups.append(GroupedEvidence(
            group_id=f"{source}:{event}",
            source=source,
            event=event,
            severity=round(float(np.clip(severity, 0, 1)), 4),
            confidence=round(float(np.clip(combined_confidence, 0, 1)), 4),
            source_weight=config.source_weights.get(source, config.unknown_source_weight),
            item_count=len(ordered),
            origin_ids=ids,
            explanations=[item.explanation for _, item in ordered if item.explanation],
        ))
    return sorted(groups, key=lambda group: (group.source, group.event))
