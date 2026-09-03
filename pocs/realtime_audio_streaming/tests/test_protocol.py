import pytest

from realtime_audio.protocol import ProtocolError, parse_control_message, validate_start
from realtime_audio.schemas import StartMessage, StopMessage


def test_valid_start_and_stop_messages():
    start = parse_control_message(
        '{"type":"start","sample_rate":16000,"channels":1,"sample_format":"pcm_s16le"}'
    )
    assert isinstance(start, StartMessage)
    validate_start(start)
    assert isinstance(parse_control_message('{"type":"stop"}'), StopMessage)


@pytest.mark.parametrize("payload", ["not json", "[]", '{"type":"other"}'])
def test_malformed_messages(payload):
    with pytest.raises(ProtocolError):
        parse_control_message(payload)


def test_unsupported_format_rejected():
    message = StartMessage(type="start", sample_rate=44100, channels=2, sample_format="float32")
    with pytest.raises(ProtocolError):
        validate_start(message)
