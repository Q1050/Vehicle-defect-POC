from audio_diagnostics.features import AudioFeatureExtractor
from audio_diagnostics.preprocessing import AudioPreprocessor
from conftest import wav_bytes


def test_feature_summary_is_compact_and_finite(clean_tone):
    audio = AudioPreprocessor().process(wav_bytes(clean_tone), "tone.wav")
    result = AudioFeatureExtractor().extract(audio)
    assert len(result.public.mfcc_summary) == 13
    assert set(result.public.frequency_band_energy) == {"low", "mid", "high"}
    assert all(0 <= value <= 1 for value in result.signals.values())
