"""Optional speech transcription contract; disabled by default."""

from abc import ABC, abstractmethod

from .preprocessing import ProcessedAudio


class AudioTranscriber(ABC):
    @abstractmethod
    def transcribe(self, audio: ProcessedAudio) -> str | None:
        """Return spoken context, or None when no speech is available."""


class DisabledTranscriber(AudioTranscriber):
    def transcribe(self, audio: ProcessedAudio) -> None:
        return None


class WhisperTranscriber(AudioTranscriber):
    """Optional adapter requiring the separately installed openai-whisper package."""

    def __init__(self, model_name: str = "tiny"):
        try:
            import whisper
        except ImportError as exc:
            raise RuntimeError("Install openai-whisper and FFmpeg to enable transcription.") from exc
        self.model = whisper.load_model(model_name)

    def transcribe(self, audio: ProcessedAudio) -> str | None:
        result = self.model.transcribe(audio.samples, fp16=False)
        text = (result.get("text") or "").strip()
        return text or None
