from realtime_audio.schemas import LiveEvent
from realtime_audio.session import RealtimeAudioSession


def event(detected=True, confidence=0.8):
    return LiveEvent(
        event="possible_knocking",
        detected=detected,
        confidence=confidence,
        severity=0.6,
    )


def test_start_and_end_lifecycle():
    session = RealtimeAudioSession("Ticking sound")
    assert not session.closed
    assert session.user_description == "Ticking sound"
    session.close()
    assert session.closed


def test_two_windows_required_and_duplicate_lifecycle_is_stable():
    session = RealtimeAudioSession()
    first, first_evidence = session.stabilize([event()])
    second, second_evidence = session.stabilize([event()])
    third, third_evidence = session.stabilize([event()])
    assert first[0].state == "inactive"
    assert not first_evidence
    assert second[0].state == "started"
    assert second_evidence[0].event == "possible_knocking"
    assert third[0].state == "ongoing"
    assert len(third_evidence) == 1


def test_two_negative_windows_end_event():
    session = RealtimeAudioSession()
    session.stabilize([event()])
    session.stabilize([event()])
    first_negative, _ = session.stabilize([event(False, 0.2)])
    second_negative, evidence = session.stabilize([event(False, 0.1)])
    assert first_negative[0].state == "ongoing"
    assert second_negative[0].state == "ended"
    assert not evidence
