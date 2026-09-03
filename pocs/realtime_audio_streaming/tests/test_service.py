from types import SimpleNamespace

import pytest

from audio_diagnostics.service import AudioDiagnosticService
from realtime_audio.encoding import float32_to_pcm_s16le
from realtime_audio.service import RealtimeAudioService


class StubAnalyzer:
    def __init__(self):
        self.calls = []

    def analyze(self, audio_bytes, filename, user_description=None):
        self.calls.append((audio_bytes, filename, user_description))
        detected = len(self.calls) >= 1
        return SimpleNamespace(
            overall=SimpleNamespace(anomaly_detected=detected, anomaly_score=0.7, confidence=0.65),
            events=[SimpleNamespace(
                event="possible_knocking",
                detected=detected,
                confidence=0.75,
                severity=0.6,
                evidence={"transient_score": 0.8},
            )],
            summary="Synthetic test result.",
        )


def test_audio_before_start_is_rejected():
    service = RealtimeAudioService(analyzer=StubAnalyzer())
    with pytest.raises(ValueError, match="Unknown"):
        service.push_audio("missing", b"\x00\x00")


def test_multiple_windows_and_session_cleanup(sine_samples):
    analyzer = StubAnalyzer()
    service = RealtimeAudioService(analyzer=analyzer)
    started = service.start_session("Ticking")
    updates = service.push_audio(started.session_id, float32_to_pcm_s16le(sine_samples))
    assert len(updates) == 3
    assert [item.window.start_seconds for item in updates] == [0, 1, 2]
    assert analyzer.calls[0][2] == "Ticking"
    assert updates[1].stabilized_events[0].state == "started"
    assert updates[2].stabilized_events[0].state == "ongoing"
    assert updates[1].evidence[0].source == "audio"
    ended = service.end_session(started.session_id)
    assert ended.windows_processed == 3
    assert started.session_id not in service.sessions


def test_existing_audio_diagnostic_service_adapter(sine_samples):
    service = RealtimeAudioService(analyzer=AudioDiagnosticService())
    started = service.start_session()
    four_seconds = sine_samples[:64_000]
    updates = service.push_audio(started.session_id, float32_to_pcm_s16le(four_seconds))
    assert len(updates) == 1
    assert len(updates[0].events) == 4
    assert updates[0].model_dump()["type"] == "diagnostic_update"
    service.end_session(started.session_id)


def test_disconnect_cleans_up():
    service = RealtimeAudioService(analyzer=StubAnalyzer())
    started = service.start_session()
    service.disconnect(started.session_id)
    assert not service.sessions
