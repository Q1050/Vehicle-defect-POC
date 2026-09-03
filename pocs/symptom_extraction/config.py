"""Central configuration for the standalone symptom-extraction POC."""

MAX_TEXT_LENGTH = 20_000
BASE_CONFIDENCE = 0.92
UNCERTAIN_CONFIDENCE_FACTOR = 0.68
NEGATION_CONTEXT_CHARS = 55
MODIFIER_CONTEXT_CHARS = 35

SEVERITY_MODIFIERS = {
    "slight": 0.25,
    "slightly": 0.25,
    "tiny": 0.2,
    "small": 0.3,
    "minor": 0.3,
    "bad": 0.7,
    "badly": 0.7,
    "loud": 0.7,
    "very loud": 0.85,
    "severe": 0.85,
    "severely": 0.85,
    "violent": 0.95,
    "violently": 0.95,
    "lots of": 0.85,
    "a lot of": 0.85,
}

DEFAULT_SEVERITY = 0.5
