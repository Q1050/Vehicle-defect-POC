import sys
from pathlib import Path

import pytest

POC_ROOT = Path(__file__).resolve().parents[1]
if str(POC_ROOT) not in sys.path:
    sys.path.insert(0, str(POC_ROOT))

from symptom_extraction.service import SymptomExtractionService


@pytest.fixture
def service() -> SymptomExtractionService:
    return SymptomExtractionService()
