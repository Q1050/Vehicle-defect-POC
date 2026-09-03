# Audio Diagnostic POC

This internship proof of concept accepts a short vehicle audio recording, measures acoustic characteristics with Librosa/NumPy, and returns cautious heuristic anomaly events and confidence scores. It is a learning artifact, not a definitive mechanical diagnostic tool.

## Architecture

~~~text
FastAPI app.py
    -> AudioDiagnosticService
        -> AudioPreprocessor
        -> AudioFeatureExtractor
        -> AudioEventClassifier
        -> AnomalyEngine
        -> deterministic summary + structured result
~~~

app.py only translates HTTP input/errors. The audio_diagnostics package has no FastAPI or UI dependency and can be imported without starting a server.

~~~text
pocs/audio_diagnostics/
|-- app.py
|-- config.py
|-- requirements.txt
|-- audio_diagnostics/
|   |-- service.py
|   |-- preprocessing.py
|   |-- features.py
|   |-- anomaly.py
|   |-- classifier.py
|   |-- transcription.py
|   |-- summaries.py
|   |-- schemas.py
|-- tests/
|-- scripts/generate_test_audio.py
|-- samples/
|-- outputs/
|-- models/
~~~

## Processing pipeline

1. Validate WAV, MP3, M4A, or FLAC, non-empty bytes, and duration between 0.5 and 30 seconds.
2. Decode with SoundFile, average channels to mono, and resample to 22,050 Hz.
3. Trim leading/trailing silence at 35 dB below peak and normalize peak amplitude to 0.95.
4. Extract compact statistics for RMS, zero-crossing rate, spectral centroid, bandwidth, rolloff, 13 MFCC means/stds, onset strength, temporal RMS variation, and low/mid/high band-energy ratios.
5. Derive normalized measured signals and pass them to the swappable classifier and anomaly engine.
6. Return measured features separately from interpreted events, a deterministic summary, and evidence-fusion records.

MP3 and M4A decoding depends on the codecs available to libsndfile. If they do not decode on your system, install FFmpeg and convert to WAV/FLAC before upload. WAV is the most reliable demo format.

## Current heuristic implementation

All thresholds and weights are centralized in config.py. Event detection uses a score threshold of 0.58:

- possible_knocking = 40% transient score + 35% low/mid-band energy + 25% impulse-rate score.
- possible_hissing = 45% high-frequency energy + 30% spectral flatness + 25% high-frequency persistence.
- possible_misfire_pattern = 45% onset-interval irregularity + 35% temporal RMS variation + 20% pulse activity.
- possible_bearing_noise = 35% tonal concentration + 25% mid-band energy + 25% energy persistence + 15% repetitive component.

Severity maps the score above a low evidence floor into 0–1. The overall anomaly score is 75% of the strongest event score plus 25% of general signal activity. Overall confidence combines event strength with distance from the ambiguous midpoint. These are explainable engineering heuristics, not learned probabilities or calibrated diagnostic accuracy.

The AudioEventClassifier and AnomalyEngine abstract interfaces allow a future trained classifier, Isolation Forest, One-Class SVM, autoencoder, or embedding model to replace the heuristics without changing the service or API contract. No trained artifact is currently stored in models/.

## Optional Whisper support

Engine-noise analysis does not require speech. DisabledTranscriber is the default. WhisperTranscriber provides an optional local interface, but openai-whisper and FFmpeg are deliberately not included in the core requirements. Transcribed speech is contextual output and is not used to alter acoustic measurements.

## API

Start from this directory:

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app:app --reload
~~~

- GET /health returns service status.
- GET /docs opens interactive OpenAPI documentation.
- POST /api/v1/diagnostics/audio accepts multipart form fields audio (required) and user_description (optional).

Example:

~~~powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/diagnostics/audio -F "audio=@samples/synthetic_impulses.wav" -F "user_description=There is a ticking sound when I accelerate."
~~~

Response shape:

~~~json
{
  "status": "completed",
  "audio": {"filename": "synthetic_impulses.wav", "duration_seconds": 4.0, "sample_rate": 22050, "samples_analyzed": 88200},
  "overall": {"anomaly_detected": true, "anomaly_score": 0.7, "confidence": 0.65},
  "events": [{"event": "possible_knocking", "detected": true, "confidence": 0.71, "severity": 0.61, "evidence": {"transient_score": 0.8}}],
  "features": {"rms": {"mean": 0.1, "std": 0.02, "max": 0.8}, "mfcc_summary": [], "measured_signals": {}},
  "transcription": null,
  "summary": "Cautious local interpretation...",
  "limitations": ["This is an explainable heuristic POC..."]
}
~~~

The actual response includes all four interpreted event records and all documented compact feature summaries.

## Python use and evidence compatibility

~~~python
from audio_diagnostics.service import AudioDiagnosticService

service = AudioDiagnosticService()
result = service.analyze(audio_bytes=audio_bytes, filename="engine.wav", user_description="There is a ticking sound when I accelerate.")
evidence = result.to_evidence()
~~~

Each evidence record has the stable shape:

~~~json
{"source": "audio", "event": "possible_knocking", "severity": 0.58, "confidence": 0.71}
~~~

For a later shared backend, copy/import the package plus config.py, or package them together as a backend audio service. The mobile app should call that shared backend, not this POC server.

## Tests and synthetic fixtures

~~~powershell
python scripts\generate_test_audio.py
pytest -q
python -m compileall -q app.py config.py audio_diagnostics tests
~~~

Synthetic clean tone, impulses, high-frequency noise, and irregular pulses exercise plumbing and relative heuristic behavior. They do not reproduce or validate real vehicle faults.

## Known limitations

- No real-world labeled vehicle-audio dataset has been used to tune or validate thresholds.
- Scores are heuristic similarities, not calibrated probabilities.
- A clean sine wave is only a software test and is not a definition of a healthy engine.
- Microphone placement, automatic gain control, wind, speech, road noise, RPM, load, and vehicle type may dominate results.
- Hiss, knocks, irregular timing, and tonal energy can have benign or unrelated causes.
- Silence trimming can remove meaningful very quiet context.
- Phone audio may omit frequencies used by a heuristic.
- The service must not be used to make safety or repair decisions without physical inspection.

## Real-data validation direction

Collect consented, de-identified WAV recordings from multiple vehicles and phones: healthy baselines and mechanic-confirmed examples of knock-like transients, vacuum/air leaks, misfire at idle and under load, accessory/bearing noises, and confusing negatives such as injector tick, fans, rain, speech, wind, and road noise. Record consistent metadata: vehicle, engine, RPM/load, microphone position/distance, environment, confirmed cause, and repair outcome.
