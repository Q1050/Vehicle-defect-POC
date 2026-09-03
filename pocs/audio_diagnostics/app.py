"""Thin FastAPI adapter for the reusable audio diagnostics service."""

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from audio_diagnostics.preprocessing import AudioValidationError
from audio_diagnostics.schemas import AudioDiagnosticResult
from audio_diagnostics.service import AudioDiagnosticService
from config import AUDIO_CONFIG


app = FastAPI(
    title="Audio Diagnostic POC",
    version="0.1.0",
    description="Learning-focused acoustic anomaly heuristics for vehicle recordings.",
)
service = AudioDiagnosticService()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "audio-diagnostics-poc",
        "classifier": "heuristic",
        "transcription_enabled": False,
    }


@app.post("/api/v1/diagnostics/audio", response_model=AudioDiagnosticResult)
async def analyze_audio(
    audio: UploadFile = File(...),
    user_description: str | None = Form(default=None),
) -> AudioDiagnosticResult:
    filename = Path(audio.filename or "upload").name
    extension = Path(filename).suffix.lower()
    if extension not in AUDIO_CONFIG.supported_extensions:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported format. Supported: {', '.join(AUDIO_CONFIG.supported_extensions)}.",
        )
    try:
        audio_bytes = await audio.read()
        return service.analyze(audio_bytes, filename, user_description)
    except AudioValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        await audio.close()
