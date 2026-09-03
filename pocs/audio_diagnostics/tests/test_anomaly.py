from audio_diagnostics.anomaly import HeuristicAnomalyEngine
from audio_diagnostics.classifier import HeuristicAudioEventClassifier
from audio_diagnostics.features import AudioFeatureExtractor
from audio_diagnostics.preprocessing import AudioPreprocessor
from conftest import wav_bytes


def analyze_signal(signal):
    processed = AudioPreprocessor().process(wav_bytes(signal), "test.wav")
    features = AudioFeatureExtractor().extract(processed)
    events = HeuristicAudioEventClassifier().predict(features)
    overall = HeuristicAnomalyEngine().score(features, events)
    return events, overall


def test_scores_are_clamped(repeated_impulses):
    events, overall = analyze_signal(repeated_impulses)
    assert all(0 <= event.confidence <= 1 and 0 <= event.severity <= 1 for event in events)
    assert 0 <= overall.anomaly_score <= 1
    assert 0 <= overall.confidence <= 1


def test_impulses_raise_knocking_score(repeated_impulses, clean_tone):
    impulse_events, _ = analyze_signal(repeated_impulses)
    clean_events, _ = analyze_signal(clean_tone)
    impulse_score = next(event.confidence for event in impulse_events if event.event == "possible_knocking")
    clean_score = next(event.confidence for event in clean_events if event.event == "possible_knocking")
    assert impulse_score > clean_score


def test_high_frequency_noise_raises_hissing_score(high_frequency_noise, clean_tone):
    noise_events, _ = analyze_signal(high_frequency_noise)
    clean_events, _ = analyze_signal(clean_tone)
    noise_score = next(event.confidence for event in noise_events if event.event == "possible_hissing")
    clean_score = next(event.confidence for event in clean_events if event.event == "possible_hissing")
    assert noise_score > clean_score
