"""Persistence helpers for chat sessions."""

import base64
import io
import json
import os

from PIL import Image


def _encode_bytes(value):
    return base64.b64encode(value).decode("utf-8")


def _decode_bytes(value):
    return base64.b64decode(value.encode("utf-8"))


def _serialize_attachment(attachment):
    payload = dict(attachment)
    if "bytes" in payload and isinstance(payload["bytes"], (bytes, bytearray)):
        payload["bytes"] = _encode_bytes(payload["bytes"])
    return payload


def _deserialize_attachment(attachment):
    payload = dict(attachment)
    if "bytes" in payload and isinstance(payload["bytes"], str):
        payload["bytes"] = _decode_bytes(payload["bytes"])
    return payload


def _serialize_image(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return _encode_bytes(buffer.getvalue())


def _deserialize_image(value):
    return Image.open(io.BytesIO(_decode_bytes(value))).convert("RGB")


def _serialize_report(report):
    payload = dict(report)
    if "video_bytes" in payload and isinstance(payload["video_bytes"], (bytes, bytearray)):
        payload["video_bytes"] = _encode_bytes(payload["video_bytes"])
    if "annotated_image" in payload and payload["annotated_image"] is not None:
        payload["annotated_image"] = _serialize_image(payload["annotated_image"])
    return payload


def _deserialize_report(report):
    payload = dict(report)
    if "video_bytes" in payload and isinstance(payload["video_bytes"], str):
        payload["video_bytes"] = _decode_bytes(payload["video_bytes"])
    if "annotated_image" in payload and isinstance(payload["annotated_image"], str):
        payload["annotated_image"] = _deserialize_image(payload["annotated_image"])
    return payload


def _serialize_message(message):
    payload = dict(message)
    metadata = dict(payload.get("metadata", {}))
    if "attachments" in metadata:
        metadata["attachments"] = [_serialize_attachment(item) for item in metadata["attachments"]]
    payload["metadata"] = metadata
    return payload


def _deserialize_message(message):
    payload = dict(message)
    metadata = dict(payload.get("metadata", {}))
    if "attachments" in metadata:
        metadata["attachments"] = [_deserialize_attachment(item) for item in metadata["attachments"]]
    payload["metadata"] = metadata
    return payload


def save_chat_state(file_path, chat_sessions, current_chat_id):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    payload = {
        "current_chat_id": current_chat_id,
        "chat_sessions": [
            {
                **chat_session,
                "messages": [_serialize_message(message) for message in chat_session.get("messages", [])],
                "reports": [_serialize_report(report) for report in chat_session.get("reports", [])],
            }
            for chat_session in chat_sessions
        ],
    }
    with open(file_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def load_chat_state(file_path):
    if not os.path.exists(file_path):
        return None

    with open(file_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    return {
        "current_chat_id": payload.get("current_chat_id"),
        "chat_sessions": [
            {
                **chat_session,
                "messages": [_deserialize_message(message) for message in chat_session.get("messages", [])],
                "reports": [_deserialize_report(report) for report in chat_session.get("reports", [])],
            }
            for chat_session in payload.get("chat_sessions", [])
        ],
    }
