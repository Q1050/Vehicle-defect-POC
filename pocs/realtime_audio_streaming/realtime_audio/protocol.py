"""Validation helpers for JSON control messages."""

import json

from pydantic import ValidationError

from .config import REALTIME_CONFIG, RealtimeAudioConfig
from .schemas import StartMessage, StopMessage


class ProtocolError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def parse_control_message(payload: str) -> StartMessage | StopMessage:
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ProtocolError("malformed_json", "Control messages must be valid JSON.") from exc
    if not isinstance(value, dict):
        raise ProtocolError("invalid_message", "Control messages must be JSON objects.")
    message_type = value.get("type")
    try:
        if message_type == "start":
            return StartMessage.model_validate(value)
        if message_type == "stop":
            return StopMessage.model_validate(value)
    except ValidationError as exc:
        raise ProtocolError("invalid_message", str(exc)) from exc
    raise ProtocolError("unknown_message_type", "Expected a start or stop control message.")


def validate_start(
    message: StartMessage,
    config: RealtimeAudioConfig = REALTIME_CONFIG,
) -> None:
    if message.sample_rate != config.sample_rate:
        raise ProtocolError("unsupported_sample_rate", f"V1 requires {config.sample_rate} Hz PCM.")
    if message.channels != config.channels:
        raise ProtocolError("unsupported_channels", "V1 requires mono audio.")
    if message.sample_format != config.sample_format:
        raise ProtocolError("unsupported_sample_format", f"V1 requires {config.sample_format}.")
