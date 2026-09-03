"""Thin FastAPI wrapper for VideoDiagnosticService."""

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from video_diagnostics.config import VIDEO_CONFIG
from video_diagnostics.preprocessing import VideoValidationError
from video_diagnostics.schemas import VideoDiagnosticResult
from video_diagnostics.service import VideoDiagnosticService


app = FastAPI(title="Video Event Detection POC", version="0.1.0")
service = VideoDiagnosticService()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "video-event-detection-poc", "engine": "opencv-heuristics"}


@app.post("/api/v1/diagnostics/video", response_model=VideoDiagnosticResult)
async def analyze_video(
    video: UploadFile = File(...),
    user_description: str | None = Form(default=None),
) -> VideoDiagnosticResult:
    filename = Path(video.filename or "upload").name
    if Path(filename).suffix.lower() not in VIDEO_CONFIG.supported_extensions:
        raise HTTPException(status_code=415, detail="Unsupported video format.")
    try:
        return service.analyze(await video.read(), filename, user_description)
    except VideoValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        await video.close()
