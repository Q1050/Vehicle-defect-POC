"""Text and score normalization helpers."""

import re

from config import DEFAULT_SEVERITY, MODIFIER_CONTEXT_CHARS, SEVERITY_MODIFIERS


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def clamp(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 4)


def severity_near(text: str, start: int, end: int) -> float:
    context = text[max(0, start - MODIFIER_CONTEXT_CHARS):min(len(text), end + MODIFIER_CONTEXT_CHARS)].lower()
    matches = [(len(phrase), score) for phrase, score in SEVERITY_MODIFIERS.items() if phrase in context]
    return max(matches, default=(0, DEFAULT_SEVERITY), key=lambda item: (item[1], item[0]))[1]
