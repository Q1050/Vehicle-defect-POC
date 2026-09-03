"""Stable reusable service interface."""

from config import MAX_TEXT_LENGTH
from .evidence import observations_to_evidence
from .extractor import RuleBasedSymptomExtractor, SymptomExtractor
from .normalization import normalize_text
from .schemas import SymptomExtractionResult
from .summaries import build_summary


class SymptomExtractionService:
    def __init__(self, extractor: SymptomExtractor | None = None) -> None:
        self.extractor = extractor or RuleBasedSymptomExtractor()

    def extract(self, text: str) -> SymptomExtractionResult:
        normalized = normalize_text(text)
        if not normalized:
            raise ValueError("Symptom description must not be empty.")
        if len(normalized) > MAX_TEXT_LENGTH:
            raise ValueError(f"Symptom description must not exceed {MAX_TEXT_LENGTH} characters.")
        symptoms, conditions, temporal = self.extractor.extract(normalized)
        return SymptomExtractionResult(
            original_text=normalized,
            symptoms=symptoms,
            conditions=conditions,
            temporal=temporal,
            evidence=observations_to_evidence(symptoms, conditions),
            summary=build_summary(symptoms, conditions, temporal),
            limitations=[
                "Rule-based extraction may miss uncommon wording or complex references.",
                "User reports are observations, not verified diagnoses.",
            ],
        )
