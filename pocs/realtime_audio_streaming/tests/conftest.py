from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_ROOT = PROJECT_ROOT.parent / "audio_diagnostics"
for path in (str(AUDIO_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


@pytest.fixture
def sine_samples():
    sample_rate = 16_000
    t = np.arange(sample_rate * 6) / sample_rate
    return (0.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
