"""Thin FastAPI/WebSocket adapter for realtime audio sessions."""

from pathlib import Path
import sys

from fastapi import FastAPI, WebSocket, WebSocketDisconnect


# Import the completed sibling Audio POC in its native self-contained context.
AUDIO_POC_ROOT = Path(__file__).resolve().parents[1] / "audio_diagnostics"
sys.path.insert(0, str(AUDIO_POC_ROOT))
from audio_diagnostics.service import AudioDiagnosticService  # noqa: E402
sys.path.pop(0)

from realtime_audio.config import REALTIME_CONFIG  # noqa: E402
from realtime_audio.protocol import ProtocolError, parse_control_message, validate_start  # noqa: E402
from realtime_audio.schemas import ErrorEvent, StartMessage, StopMessage  # noqa: E402
from realtime_audio.service import RealtimeAudioService  # noqa: E402


app = FastAPI(title="Realtime Audio Streaming POC", version="0.1.0")
service = RealtimeAudioService(analyzer=AudioDiagnosticService())


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "realtime-audio-streaming-poc",
        "transport": "websocket",
        "audio_format": REALTIME_CONFIG.sample_format,
        "sample_rate": REALTIME_CONFIG.sample_rate,
    }


async def send_error(websocket: WebSocket, code: str, message: str) -> None:
    await websocket.send_json(ErrorEvent(code=code, message=message).model_dump())


@app.websocket("/ws/audio")
async def audio_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    session_id = None
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            text = message.get("text")
            audio = message.get("bytes")

            if text is not None:
                try:
                    control = parse_control_message(text)
                    if isinstance(control, StartMessage):
                        if session_id is not None:
                            await send_error(websocket, "session_already_started", "A session is already active.")
                            continue
                        validate_start(control)
                        started = service.start_session(control.user_description)
                        session_id = started.session_id
                        await websocket.send_json(started.model_dump())
                        continue
                    if isinstance(control, StopMessage):
                        if session_id is None:
                            await send_error(websocket, "session_not_started", "Start a session before stopping it.")
                            continue
                        ended = service.end_session(session_id)
                        session_id = None
                        await websocket.send_json(ended.model_dump())
                        await websocket.close(code=1000)
                        return
                except ProtocolError as exc:
                    await send_error(websocket, exc.code, str(exc))
                continue

            if audio is not None:
                if session_id is None:
                    await send_error(websocket, "session_not_started", "Send a valid start message before audio.")
                    continue
                try:
                    for update in service.push_audio(session_id, audio):
                        await websocket.send_json(update.model_dump())
                except ValueError as exc:
                    await send_error(websocket, "invalid_audio", str(exc))
                continue

            await send_error(websocket, "empty_message", "Expected a JSON control message or binary PCM chunk.")
    except WebSocketDisconnect:
        pass
    finally:
        service.disconnect(session_id)
