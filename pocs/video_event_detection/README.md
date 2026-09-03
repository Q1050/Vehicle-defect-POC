# Video Event Detection POC

This internship POC analyzes short vehicle videos for unstable localized motion and persistent smoke-like visual patterns. It combines a reusable local OpenCV service with a thin FastAPI adapter and is also used by the existing Streamlit attachment flow.

It is a software and learning POC, not a validated mechanical diagnostic system.

## Reused existing behavior

The main project already accepted MP4, MOV, AVI, MKV, and WEBM uploads, stored video bytes with chat messages, rendered uploaded videos, and optionally uploaded the full video to Gemini. That upload/chat/Gemini path remains in place. Previously, it did not extract frames or run any local video analysis.

## Architecture

~~~text
Streamlit attachment flow          FastAPI POC
             \                       /
              -> VideoDiagnosticService
                   -> VideoFrameSampler
                   -> VibrationAnalyzer
                   -> SmokeAnalyzer
                   -> deterministic summary
                   -> Pydantic result / evidence records

Optional Gemini full-video interpretation receives local findings as context.
~~~

Core logic lives in video_diagnostics and has no Streamlit or FastAPI dependency.

## Local processing

VideoFrameSampler writes bytes to a temporary file because OpenCV VideoCapture requires a file/stream source. It validates the extension and stream metadata, rejects empty/unreadable media, enforces a 30-second maximum, and requires at least three sampled frames.

Frames are sampled every 0.2 seconds (approximately 5 FPS) and resized to at most 640 pixels wide for analysis. The result includes original FPS, width, height, frame count, duration, analyzed-frame count, actual sampling interval, and timestamps.

## Vibration heuristic

For every sampled frame pair:

1. Convert frames to grayscale.
2. Calculate dense Farneback optical flow.
3. Detect up to 250 Shi-Tomasi feature points and track them with pyramidal Lucas-Kanade flow.
4. Use the median tracked displacement as broad camera translation.
5. Subtract that translation from dense flow.
6. Exclude dilated, plausibly sized low-texture smoke-candidate regions when masks are supplied by the service.
7. Measure residual flow only in the remaining textured regions, reducing flat-region and plume-motion bias.
8. Calculate mean residual motion, residual variance, localized active fraction, short-lag autocorrelation, global motion, and camera-motion dominance.

The confidence score is:

~~~text
(45% residual magnitude
 + 25% residual variance
 + 20% localized-motion fraction
 + 10% repetitive-motion score)
 x camera-motion discount
~~~

The candidate threshold is 0.40. All normalization constants and thresholds are centralized in video_diagnostics/config.py.

## Smoke heuristic

Each sampled frame is converted to HSV and grayscale. Candidate pixels must have low saturation, medium/high brightness, and low blurred Laplacian texture. Morphological opening removes small noise.

The heuristic rejects candidate coverage below 1% or above 55% of the frame, then measures plausible area, frame persistence, mask overlap, mask changes/spreading, and upward centroid movement.

~~~text
30% candidate area
+ 30% frame persistence
+ 20% temporal overlap
+ 15% spreading/change
+ 5% upward movement
~~~

The smoke candidate threshold is 0.52. White walls, steam, reflections, exposure changes, and moving pale objects can still cause false positives.

## API

From this directory:

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\generate_test_videos.py
uvicorn app:app --reload
~~~

Endpoints:

- GET /health
- GET /docs
- POST /api/v1/diagnostics/video

The POST endpoint accepts multipart fields video (required) and user_description (optional).

~~~powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/diagnostics/video -F "video=@samples/synthetic_vibration.mp4" -F "user_description=The engine appears to shake."
~~~

Results contain metadata, possible_engine_vibration and possible_smoke events, confidence/severity/evidence, a deterministic summary, and limitations.

## Evidence compatibility

~~~python
from video_diagnostics.service import VideoDiagnosticService

result = VideoDiagnosticService().analyze(video_bytes, "engine.mp4", "Visible shaking")
evidence = result.to_evidence()
~~~

Normalized record:

~~~json
{"source": "video", "event": "possible_engine_vibration", "severity": 0.61, "confidence": 0.72}
~~~

## Tests

~~~powershell
python scripts\generate_test_videos.py
pytest -q
python -m compileall -q app.py video_diagnostics tests scripts
~~~

Synthetic clips cover static imagery, global translation, a localized oscillating rectangle, and a moving light plume. They validate software behavior only and do not represent real engine vibration or smoke.

## Gemini role

The existing whole-video Gemini upload is preserved in the main application. Local OpenCV analysis always runs first. When Gemini is configured, its prompt includes the user description and local event findings, and the full video remains available to Gemini for optional interpretation. Without Gemini, metadata, local events, evidence, and the deterministic summary remain available.

## Limitations

- No real labeled vehicle-video set was used to tune or validate scores.
- Optical flow cannot determine mechanical cause.
- Handheld motion, autofocus, rolling shutter, fans, belts, shadows, and compression can produce residual motion.
- Smoke-mask exclusion reduces plume cross-signal motion, but incomplete or incorrect masks can still affect vibration scores.
- Steam, dust, glare, pale moving objects, and backgrounds can resemble smoke.
- Sampling may miss brief events.
- OpenCV codec support varies by operating system and installation.
- Frame-level YOLO is not run because the trained image model classes do not include vibration or smoke.

For a shared mobile backend, move/import video_diagnostics as the video service. Keep the POC FastAPI app, synthetic fixtures, and Streamlit-specific adapter outside the reusable service.
