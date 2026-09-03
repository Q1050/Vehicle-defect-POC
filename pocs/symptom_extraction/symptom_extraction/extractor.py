"""Replaceable extraction interface and deterministic baseline implementation."""

from abc import ABC, abstractmethod
import re

from config import BASE_CONFIDENCE, UNCERTAIN_CONFIDENCE_FACTOR
from .negation import is_negated, is_third_party, is_uncertain
from .normalization import clamp, severity_near
from .schemas import ConditionObservation, SymptomObservation, TemporalInformation
from .vocabulary import CONDITION_ALIASES, SYMPTOM_ALIASES


class SymptomExtractor(ABC):
    @abstractmethod
    def extract(self, text: str) -> tuple[list[SymptomObservation], list[ConditionObservation], TemporalInformation]:
        """Extract observations without diagnosing causes."""


class RuleBasedSymptomExtractor(SymptomExtractor):
    def extract(self, text: str) -> tuple[list[SymptomObservation], list[ConditionObservation], TemporalInformation]:
        symptoms: dict[str, SymptomObservation] = {}
        for event, patterns in SYMPTOM_ALIASES.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.I):
                    if is_third_party(text, match.start(), match.end()):
                        continue
                    uncertain = is_uncertain(text, match.start())
                    observation = SymptomObservation(
                        event=event,
                        text=match.group(0).strip(),
                        confidence=clamp(BASE_CONFIDENCE * (UNCERTAIN_CONFIDENCE_FACTOR if uncertain else 1)),
                        severity=severity_near(text, match.start(), match.end()),
                        negated=is_negated(text, match.start(), match.end()),
                        uncertain=uncertain,
                    )
                    current = symptoms.get(event)
                    if current is None or self._rank(observation) > self._rank(current):
                        symptoms[event] = observation

        conditions: dict[str, ConditionObservation] = {}
        for condition, patterns in CONDITION_ALIASES.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.I)
                if match and not is_third_party(text, match.start(), match.end()):
                    conditions.setdefault(condition, ConditionObservation(
                        condition=condition, text=match.group(0).strip(), confidence=0.9,
                    ))

        return list(symptoms.values()), list(conditions.values()), self._temporal(text)

    @staticmethod
    def _rank(item: SymptomObservation) -> tuple[float, int, int]:
        return item.severity, len(item.text), int(not item.uncertain)

    @staticmethod
    def _temporal(text: str) -> TemporalInformation:
        def phrase(pattern: str) -> str | None:
            match = re.search(pattern, text, re.I)
            return match.group(0).strip() if match else None

        onset = phrase(r"\b(?:(?:started|began|first noticed) |came on )(?:today|yesterday|last (?:night|week|month)|\d+ (?:days?|weeks?|months?) ago)\b")
        duration = phrase(r"\b(?:for|over) (?:about |around )?(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+) (?:minutes?|hours?|days?|weeks?|months?|years?)\b")
        frequency = phrase(r"\b(?:every (?:morning|day|time|week)|occasionally|sometimes|intermittently|constantly|always|only happens sometimes)\b")
        trend: str | None = None
        if re.search(r"\b(?:getting|gets?|becoming) (?:worse|louder|stronger|more frequent)\b", text, re.I):
            trend = "worsening"
        elif re.search(r"\b(?:gets? better|improves?|goes? away|gets? smoother) (?:after|when|once).*(?:warm|driv)", text, re.I):
            trend = "improves_when_warm"
        elif re.search(r"\b(?:getting|gets?|becoming) (?:better|quieter|smoother|less frequent)\b", text, re.I):
            trend = "improving"
        return TemporalInformation(onset=onset, duration=duration, frequency=frequency, trend=trend)
