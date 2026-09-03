"""Small contextual rules for negation, uncertainty, and non-user reports."""

import re

NEGATION_RE = re.compile(r"\b(?:no|not|isn't|aren't|wasn't|weren't|don't|doesn't|didn't|cannot|can't|without|never)\b", re.I)
UNCERTAINTY_RE = re.compile(r"\b(?:maybe|I think|might|possibly|seems? like|not sure|could be)\b", re.I)
THIRD_PARTY_RE = re.compile(r"\b(?:my friend|his car|her car|their car|I read online|online says?|people say)\b", re.I)


def sentence_span(text: str, start: int, end: int) -> str:
    left = max(text.rfind(".", 0, start), text.rfind("!", 0, start), text.rfind("?", 0, start)) + 1
    stops = [position for mark in ".!?" if (position := text.find(mark, end)) >= 0]
    right = min(stops) + 1 if stops else len(text)
    return text[left:right]


def is_third_party(text: str, start: int, end: int) -> bool:
    return bool(THIRD_PARTY_RE.search(sentence_span(text, start, end)))


def is_negated(text: str, start: int, end: int, window: int = 55) -> bool:
    before = text[max(0, start - window):start]
    sentence = sentence_span(text, start, end)
    if NEGATION_RE.search(before):
        return True
    # Handles: "checked for an oil leak but didn't find one."
    return bool(re.search(r"\b(?:didn't|did not|couldn't|could not) find (?:one|any|it)\b", sentence, re.I))


def is_uncertain(text: str, start: int, window: int = 55) -> bool:
    return bool(UNCERTAINTY_RE.search(text[max(0, start - window):start]))
