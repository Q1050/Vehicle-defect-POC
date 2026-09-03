"""Framework-independent video diagnostic orchestration."""

from .motion import VibrationAnalyzer
from .preprocessing import VideoFrameSampler
from .schemas import VideoDiagnosticResult
from .smoke import SmokeAnalyzer
from .summaries import build_summary


LIMITATIONS = [
    "The event scores are uncalibrated heuristics and have not been validated on real vehicle faults.",
    "Handheld motion, autofocus, compression, lighting, steam, and reflections can change the scores.",
    "Only sampled frames are analyzed; short events between samples may be missed.",
    "A candidate event is acoustic/visual similarity, not a confirmed mechanical cause.",
]


class VideoDiagnosticService:
    def __init__(
        self,
        frame_sampler: VideoFrameSampler | None = None,
        vibration_analyzer: VibrationAnalyzer | None = None,
        smoke_analyzer: SmokeAnalyzer | None = None,
    ):
        self.frame_sampler = frame_sampler or VideoFrameSampler()
        self.vibration_analyzer = vibration_analyzer or VibrationAnalyzer()
        self.smoke_analyzer = smoke_analyzer or SmokeAnalyzer()

    def analyze(
        self,
        video_bytes: bytes,
        filename: str,
        user_description: str | None = None,
    ) -> VideoDiagnosticResult:
        sampled = self.frame_sampler.sample(video_bytes, filename)
        smoke_masks = self.smoke_analyzer.candidate_masks(sampled.frames)
        events = [
            self.vibration_analyzer.analyze(sampled.frames, exclusion_masks=smoke_masks),
            self.smoke_analyzer.analyze(sampled.frames, candidate_masks=smoke_masks),
        ]
        return VideoDiagnosticResult(
            video=sampled.metadata,
            events=events,
            summary=build_summary(events, user_description),
            limitations=LIMITATIONS,
        )
