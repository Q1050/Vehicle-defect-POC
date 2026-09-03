"""Reusable intelligent evidence-fusion package."""

from .service import EvidenceFusionService
from .schemas import EvidenceItem, FusionResult
from .normalization import image_detections_to_evidence

__all__ = ["EvidenceFusionService", "EvidenceItem", "FusionResult", "image_detections_to_evidence"]
