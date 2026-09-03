"""Reusable session manager and AudioDiagnosticService adapter."""

from typing import Protocol, Any

from .config import REALTIME_CONFIG, RealtimeAudioConfig
from .encoding import pcm_s16le_to_float32, samples_to_wav_bytes
from .schemas import (
    DiagnosticUpdate,
    LiveEvent,
    SessionEnded,
    SessionStarted,
    WindowRange,
)
from .session import RealtimeAudioSession


class AudioAnalyzer(Protocol):
    def analyze(
        self, audio_bytes: bytes, filename: str, user_description: str | None = None
    ) -> Any: ...


def create_default_audio_analyzer() -> AudioAnalyzer:
    """Import the existing sibling POC after its directory is on sys.path."""
    from audio_diagnostics.service import AudioDiagnosticService

    return AudioDiagnosticService()


class RealtimeAudioService:
    def __init__(
        self,
        analyzer: AudioAnalyzer | None = None,
        config: RealtimeAudioConfig = REALTIME_CONFIG,
    ):
        self.analyzer = analyzer or create_default_audio_analyzer()
        self.config = config
        self.sessions: dict[str, RealtimeAudioSession] = {}

    def start_session(self, user_description: str | None = None) -> SessionStarted:
        session = RealtimeAudioSession(user_description, self.config)
        self.sessions[session.session_id] = session
        return SessionStarted(session_id=session.session_id)

    def get_session(self, session_id: str) -> RealtimeAudioSession:
        try:
            return self.sessions[session_id]
        except KeyError as exc:
            raise ValueError("Unknown or ended session.") from exc

    def push_audio(self, session_id: str, pcm_bytes: bytes) -> list[DiagnosticUpdate]:
        session = self.get_session(session_id)
        if len(pcm_bytes) > self.config.max_chunk_bytes:
            raise ValueError(f"Audio chunk exceeds {self.config.max_chunk_bytes} bytes.")
        samples = pcm_s16le_to_float32(pcm_bytes)
        if (
            session.buffer.total_samples_received + len(samples)
            > self.config.max_session_seconds * self.config.sample_rate
        ):
            raise ValueError(
                f"Session exceeds the {self.config.max_session_seconds:.0f}-second POC limit."
            )
        session.add_samples(samples)
        return self.process_available_windows(session_id)

    def process_available_windows(self, session_id: str) -> list[DiagnosticUpdate]:
        session = self.get_session(session_id)
        updates = []
        for window in session.buffer.pop_available_windows():
            wav_bytes = samples_to_wav_bytes(window.samples, self.config.sample_rate)
            result = self.analyzer.analyze(
                audio_bytes=wav_bytes,
                filename=f"stream-{session.session_id}-{session.windows_processed + 1}.wav",
                user_description=session.user_description,
            )
            events = [
                LiveEvent(
                    event=item.event,
                    detected=item.detected,
                    confidence=item.confidence,
                    severity=item.severity,
                    evidence=item.evidence,
                )
                for item in result.events
            ]
            stabilized, evidence = session.stabilize(events)
            session.windows_processed += 1
            session.latest_result = result
            updates.append(DiagnosticUpdate(
                session_id=session.session_id,
                window=WindowRange(
                    start_seconds=round(window.start_seconds, 3),
                    end_seconds=round(window.end_seconds, 3),
                ),
                overall={
                    "anomaly_detected": result.overall.anomaly_detected,
                    "anomaly_score": result.overall.anomaly_score,
                    "confidence": result.overall.confidence,
                },
                events=events,
                stabilized_events=stabilized,
                evidence=evidence,
                summary=result.summary,
            ))
        return updates

    def end_session(self, session_id: str) -> SessionEnded:
        session = self.get_session(session_id)
        session.close()
        self.sessions.pop(session_id, None)
        return SessionEnded(
            session_id=session_id,
            duration_seconds=round(session.buffer.stream_seconds, 3),
            windows_processed=session.windows_processed,
        )

    def disconnect(self, session_id: str | None) -> None:
        if session_id and session_id in self.sessions:
            self.sessions[session_id].close()
            self.sessions.pop(session_id, None)
