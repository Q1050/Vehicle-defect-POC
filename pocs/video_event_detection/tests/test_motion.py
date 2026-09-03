from video_diagnostics.service import VideoDiagnosticService
from video_diagnostics.motion import VibrationAnalyzer
from video_diagnostics.preprocessing import VideoFrameSampler


def event(result, name):
    return next(item for item in result.events if item.event == name)


def test_static_video_has_low_motion(generated_clips):
    path = generated_clips / "synthetic_static.mp4"
    result = VideoDiagnosticService().analyze(path.read_bytes(), path.name)
    vibration = event(result, "possible_engine_vibration")
    assert vibration.confidence < 0.2
    assert not vibration.detected


def test_local_oscillation_raises_vibration_above_static(generated_clips):
    service = VideoDiagnosticService()
    static_path = generated_clips / "synthetic_static.mp4"
    moving_path = generated_clips / "synthetic_vibration.mp4"
    static = event(service.analyze(static_path.read_bytes(), static_path.name), "possible_engine_vibration")
    moving = event(service.analyze(moving_path.read_bytes(), moving_path.name), "possible_engine_vibration")
    assert moving.confidence > static.confidence
    assert moving.detected


def test_global_motion_is_discounted(generated_clips):
    service = VideoDiagnosticService()
    global_path = generated_clips / "synthetic_global_motion.mp4"
    local_path = generated_clips / "synthetic_vibration.mp4"
    global_event = event(service.analyze(global_path.read_bytes(), global_path.name), "possible_engine_vibration")
    local_event = event(service.analyze(local_path.read_bytes(), local_path.name), "possible_engine_vibration")
    assert local_event.confidence > global_event.confidence
    assert global_event.confidence < 0.40
    assert not global_event.detected


def test_vibration_analyzer_still_operates_without_exclusion_masks(generated_clips):
    path = generated_clips / "synthetic_vibration.mp4"
    sampled = VideoFrameSampler().sample(path.read_bytes(), path.name)
    result = VibrationAnalyzer().analyze(sampled.frames)
    assert result.detected
