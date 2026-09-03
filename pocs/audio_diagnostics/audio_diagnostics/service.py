"""Framework-independent orchestration for audio diagnostics."""

from .anomaly import AnomalyEngine, HeuristicAnomalyEngine
from .classifier import AudioEventClassifier, HeuristicAudioEventClassifier
from .features import AudioFeatureExtractor
from .preprocessing import AudioPreprocessor
from .schemas import AudioDiagnosticResult, AudioMetadata
from .summaries import build_summary
from .transcription import AudioTranscriber, DisabledTranscriber


LIMITATIONS = [
    "This is an explainable heuristic POC, not a classifier validated on real vehicle faults.",
    "Recording position, phone processing, background noise, and engine operating state can change the scores.",
    "An interpreted event describes acoustic similarity and is not a confirmed mechanical diagnosis.",
    "Safety-critical or persistent symptoms require inspection by a qualified mechanic.",
]


class AudioDiagnosticService:
    def __init__(
        self,
        preprocessor: AudioPreprocessor | None = None,
        feature_extractor: AudioFeatureExtractor | None = None,
        classifier: AudioEventClassifier | None = None,
        anomaly_engine: AnomalyEngine | None = None,
        transcriber: AudioTranscriber | None = None,
    ):
        self.preprocessor = preprocessor or AudioPreprocessor()
        self.feature_extractor = feature_extractor or AudioFeatureExtractor()
        self.classifier = classifier or HeuristicAudioEventClassifier()
        self.anomaly_engine = anomaly_engine or HeuristicAnomalyEngine()
        self.transcriber = transcriber or DisabledTranscriber()

    def analyze(
        self,
        audio_bytes: bytes,
        filename: str,
        user_description: str | None = None,
    ) -> AudioDiagnosticResult:
        processed = self.preprocessor.process(audio_bytes, filename)
        features = self.feature_extractor.extract(processed)
        events = self.classifier.predict(features)
        overall = self.anomaly_engine.score(features, events)
        transcription = self.transcriber.transcribe(processed)
        return AudioDiagnosticResult(
            audio=AudioMetadata(
                filename=filename,
                duration_seconds=round(processed.duration_seconds, 3),
                sample_rate=processed.sample_rate,
                samples_analyzed=len(processed.samples),
            ),
            overall=overall,
            events=events,
            features=features.public,
            transcription=transcription,
            summary=build_summary(overall, events, user_description),
            limitations=LIMITATIONS,
        )
