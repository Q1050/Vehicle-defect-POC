import numpy as np

from audio_diagnostics.schemas import AudioDiagnosticResult
from audio_diagnostics.service import AudioDiagnosticService
from conftest import SAMPLE_RATE, wav_bytes


def test_service_returns_serializable_result(repeated_impulses):
    result = AudioDiagnosticService().analyze(
        wav_bytes(repeated_impulses), "engine.wav", "Ticking while accelerating."
    )
    payload = result.model_dump()
    assert payload["status"] == "completed"
    assert payload["audio"]["sample_rate"] == SAMPLE_RATE
    assert len(payload["events"]) == 4
    assert "did not change" in payload["summary"]
    assert isinstance(AudioDiagnosticResult.model_validate(payload), AudioDiagnosticResult)
    assert all(item.source == "audio" for item in result.to_evidence(detected_only=False))


def test_user_description_does_not_change_features(clean_tone):
    service = AudioDiagnosticService()
    payload = wav_bytes(clean_tone)
    first = service.analyze(payload, "tone.wav")
    second = service.analyze(payload, "tone.wav", "I hear knocking.")
    assert first.features == second.features


def test_duration_limit_rejected():
    service = AudioDiagnosticService()
    too_long = np.ones(SAMPLE_RATE * 31, dtype=np.float32) * 0.1
    try:
        service.analyze(wav_bytes(too_long), "long.wav")
    except ValueError as exc:
        assert "maximum" in str(exc)
    else:
        raise AssertionError("Expected long recording to be rejected")
