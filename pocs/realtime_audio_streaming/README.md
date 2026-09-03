# Real-Time Audio Streaming POC

This internship POC keeps a WebSocket open while raw microphone-style PCM chunks arrive, builds overlapping rolling windows, passes each complete window to the existing AudioDiagnosticService, and sends structured live results back to the client.

It validates live transport, buffering, session lifecycle, and result stabilization. It does not validate real-world vehicle diagnostic accuracy.

## Why WebSocket

Normal HTTP uploads are appropriate for complete files but do not provide a persistent bidirectional session. WebSocket lets a client stream binary chunks over one connection while the backend independently sends diagnostic updates and lifecycle/error messages. Compressed MP3/M4A uploads remain the responsibility of the normal Audio Diagnostics POC.

## Architecture

~~~text
Microphone PCM chunks
    -> FastAPI /ws/audio
    -> RealtimeAudioService
    -> RealtimeAudioSession
    -> bounded RollingAudioBuffer
    -> overlapping 4-second WAV window
    -> existing AudioDiagnosticService.analyze()
    -> current + stabilized event JSON
    -> normalized audio evidence
~~~

The WebSocket route contains transport handling only. Buffering, encoding, protocol validation, sessions, stabilization, and analysis orchestration are reusable without FastAPI WebSocket objects.

## Audio format

V1 deliberately accepts only:

~~~text
PCM signed 16-bit little-endian
Mono
16,000 Hz
Binary WebSocket frames
~~~

Unsupported rates, channel counts, and sample formats receive readable protocol errors. Arbitrary compressed chunks are not supported.

## Rolling windows

Central defaults:

~~~text
Window: 4.0 seconds
Hop:    1.0 second
~~~

This produces 0–4, 1–5, 2–6, and subsequent windows. Chunks are accumulated instead of analyzed independently. After windows are emitted, samples older than the next overlapping window start are discarded, bounding memory use. Sessions are limited to 120 seconds and individual chunks to 256,000 bytes.

Each window is encoded as an in-memory WAV and sent to AudioDiagnosticService. No temporary files are required.

## Session and stabilization

RealtimeAudioSession owns:

- A UUID session ID
- User description
- Sample and window counts
- Rolling buffer
- Latest diagnostic result
- Per-event confidence/severity history
- Positive/negative streaks
- Active event state

An event requires two consecutive detected windows before becoming active. It requires two consecutive negative windows to end. Confidence and severity are moving averages over the latest three windows.

Lifecycle values:

- inactive: current signal is not yet stable.
- started: transitioned to active in this window.
- ongoing: remains active without creating a new independent fault.
- ended: transitioned out of the active state.

The response still includes raw current-window events, so stabilization does not hide spikes. Only active stabilized events are emitted in the normalized evidence list.

## WebSocket protocol

Connect:

~~~text
ws://127.0.0.1:8000/ws/audio
~~~

Start control message:

~~~json
{
  "type": "start",
  "sample_rate": 16000,
  "channels": 1,
  "sample_format": "pcm_s16le",
  "user_description": "The engine makes a ticking sound."
}
~~~

Server:

~~~json
{"type": "session_started", "session_id": "..."}
~~~

Then send raw PCM16 as binary frames. For every available rolling window the server sends:

~~~json
{
  "type": "diagnostic_update",
  "session_id": "...",
  "window": {"start_seconds": 0.0, "end_seconds": 4.0},
  "overall": {
    "anomaly_detected": true,
    "anomaly_score": 0.68,
    "confidence": 0.64
  },
  "events": [],
  "stabilized_events": [],
  "evidence": [],
  "summary": "..."
}
~~~

Stop:

~~~json
{"type": "stop"}
~~~

Server:

~~~json
{
  "type": "session_ended",
  "session_id": "...",
  "duration_seconds": 4.0,
  "windows_processed": 1
}
~~~

Malformed controls, audio before start, duplicate starts, invalid PCM, empty chunks, oversized chunks, and unsupported configuration return type=error events where practical. Disconnects remove session state.

## Evidence output

Active stabilized events use the existing normalized contract:

~~~json
{
  "source": "audio",
  "event": "possible_knocking",
  "severity": 0.61,
  "confidence": 0.72
}
~~~

These records can later be forwarded to EvidenceFusionService. Fusion is not implemented in this POC.

The user description is passed to AudioDiagnosticService as summary context. It does not alter measured acoustic features.

## Run

From this directory:

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app:app --reload
~~~

Health and documentation:

~~~text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/docs
~~~

In a second terminal, run the fast test client:

~~~powershell
python scripts\stream_test_audio.py
~~~

To pace chunks approximately in realtime:

~~~powershell
python scripts\stream_test_audio.py --realtime
~~~

The default fixture is the existing Audio Diagnostics synthetic impulse WAV. Synthetic data validates transport only and is not a vehicle-fault recording.

## Tests

~~~powershell
python -m compileall -q app.py config.py realtime_audio tests scripts
pytest -q
~~~

Tests cover PCM conversion, overlapping windows, stale-sample cleanup, malformed controls, format rejection, session lifecycle, stabilization, ongoing/ended events, audio-before-start rejection, multiple windows, normalized evidence, real AudioDiagnosticService reuse, disconnect cleanup, and serialization.

## Flutter integration

Expected future flow:

~~~text
Flutter microphone
    -> configure PCM16 mono 16 kHz
    -> open backend WebSocket
    -> send start JSON
    -> send binary PCM chunks
    -> render diagnostic_update JSON
    -> send stop JSON
~~~

The protocol uses platform-neutral JSON and binary frames, so Flutter can implement it without changing the backend contract.

## Future speech and WebRTC paths

A later microphone pipeline can branch:

~~~text
live microphone
    |-> mechanical AudioDiagnosticService analysis
    |-> optional speech transcription / conversational assistant
~~~

No speech model or external AI service is required here.

WebRTC may later replace or complement transport for browser media tracks, NAT traversal, echo cancellation, lower-latency bidirectional voice, and conversational audio. The reusable session, buffer, WAV adapter, analyzer, stabilization, and result schemas can remain; a WebRTC track adapter would replace the WebSocket chunk ingress layer.

## Limitations

- Only PCM16 mono 16 kHz is supported.
- Processing is periodic window analysis, not sample-by-sample inference.
- AudioDiagnosticService remains a heuristic POC not validated on real faults.
- A four-second initial delay is inherent in the configured window.
- Server-side analysis can create backpressure if inference takes longer than the hop.
- No authentication, persistence, horizontal session coordination, or production rate limiting exists.
- WebSocket does not provide WebRTC media features such as echo cancellation or NAT traversal.
- Event stabilization may delay short events or keep an event active for one extra window.
- Synthetic fixtures do not establish vehicle diagnostic accuracy.
