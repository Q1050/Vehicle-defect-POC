"""Framework-independent realtime session state and event stabilization."""

from collections import defaultdict, deque
from dataclasses import dataclass
from uuid import uuid4

from .config import REALTIME_CONFIG, RealtimeAudioConfig
from .buffering import RollingAudioBuffer
from .schemas import EvidenceItem, LiveEvent, StabilizedEvent


@dataclass
class EventState:
    active: bool = False
    positive_streak: int = 0
    negative_streak: int = 0


class RealtimeAudioSession:
    def __init__(
        self,
        user_description: str | None = None,
        config: RealtimeAudioConfig = REALTIME_CONFIG,
    ):
        self.session_id = str(uuid4())
        self.user_description = user_description
        self.config = config
        self.buffer = RollingAudioBuffer(
            config.sample_rate, config.window_seconds, config.hop_seconds
        )
        self.windows_processed = 0
        self.closed = False
        self.latest_result = None
        self._history: dict[str, deque[tuple[float, float, bool]]] = defaultdict(
            lambda: deque(maxlen=config.stabilization_windows)
        )
        self._states: dict[str, EventState] = defaultdict(EventState)

    def add_samples(self, samples) -> None:
        if self.closed:
            raise RuntimeError("Session is already closed.")
        self.buffer.append(samples)

    def stabilize(
        self, events: list[LiveEvent]
    ) -> tuple[list[StabilizedEvent], list[EvidenceItem]]:
        stabilized = []
        evidence = []
        for event in events:
            history = self._history[event.event]
            history.append((event.confidence, event.severity, event.detected))
            state = self._states[event.event]
            if event.detected:
                state.positive_streak += 1
                state.negative_streak = 0
            else:
                state.negative_streak += 1
                state.positive_streak = 0

            lifecycle = "ongoing" if state.active else "inactive"
            if not state.active and state.positive_streak >= self.config.activation_consecutive_windows:
                state.active = True
                lifecycle = "started"
            elif state.active and state.negative_streak >= self.config.deactivation_consecutive_windows:
                state.active = False
                lifecycle = "ended"

            average_confidence = sum(item[0] for item in history) / len(history)
            average_severity = sum(item[1] for item in history) / len(history)
            item = StabilizedEvent(
                event=event.event,
                active=state.active,
                state=lifecycle,
                confidence=round(average_confidence, 4),
                severity=round(average_severity, 4),
            )
            stabilized.append(item)
            if state.active:
                evidence.append(EvidenceItem(
                    event=item.event,
                    severity=item.severity,
                    confidence=item.confidence,
                ))
        return stabilized, evidence

    def close(self) -> None:
        self.closed = True
