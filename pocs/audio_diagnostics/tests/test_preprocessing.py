import numpy as np
import pytest

from audio_diagnostics.preprocessing import AudioPreprocessor, AudioValidationError
from conftest import SAMPLE_RATE, wav_bytes


def test_preprocessing_normalizes_and_preserves_mono(clean_tone):
    processed = AudioPreprocessor().process(wav_bytes(clean_tone), "tone.wav")
    assert processed.sample_rate == SAMPLE_RATE
    assert processed.samples.ndim == 1
    assert np.max(np.abs(processed.samples)) == pytest.approx(0.95, abs=1e-3)


@pytest.mark.parametrize("payload, name", [(b"", "empty.wav"), (b"not audio", "bad.wav")])
def test_empty_and_unreadable_input_rejected(payload, name):
    with pytest.raises(AudioValidationError):
        AudioPreprocessor().process(payload, name)


def test_silence_rejected():
    with pytest.raises(AudioValidationError, match="silent"):
        AudioPreprocessor().process(wav_bytes(np.zeros(SAMPLE_RATE)), "silence.wav")
