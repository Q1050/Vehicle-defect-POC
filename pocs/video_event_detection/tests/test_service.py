from video_diagnostics.schemas import VideoDiagnosticResult
from video_diagnostics.service import VideoDiagnosticService


def test_service_schema_and_evidence(generated_clips):
    path = generated_clips / "synthetic_vibration.mp4"
    result = VideoDiagnosticService().analyze(
        path.read_bytes(), path.name, "The engine appears to shake."
    )
    payload = result.model_dump()
    assert payload["status"] == "completed"
    assert len(payload["events"]) == 2
    assert all(0 <= item["confidence"] <= 1 for item in payload["events"])
    assert isinstance(VideoDiagnosticResult.model_validate(payload), VideoDiagnosticResult)
    assert all(item.source == "video" for item in result.to_evidence(detected_only=False))
    assert "did not alter" in result.summary


def test_smoke_like_clip_raises_smoke_score(generated_clips):
    service = VideoDiagnosticService()
    static_path = generated_clips / "synthetic_static.mp4"
    smoke_path = generated_clips / "synthetic_smoke_like.mp4"
    static = service.analyze(static_path.read_bytes(), static_path.name)
    smoke = service.analyze(smoke_path.read_bytes(), smoke_path.name)
    static_score = next(item.confidence for item in static.events if item.event == "possible_smoke")
    smoke_event = next(item for item in smoke.events if item.event == "possible_smoke")
    vibration_event = next(item for item in smoke.events if item.event == "possible_engine_vibration")
    assert smoke_event.confidence > static_score
    assert smoke_event.detected
    assert vibration_event.confidence < 0.40
    assert not vibration_event.detected
