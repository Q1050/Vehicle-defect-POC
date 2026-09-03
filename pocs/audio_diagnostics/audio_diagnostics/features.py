"""Compact acoustic feature extraction for explainable baseline diagnostics."""

from dataclasses import dataclass

import librosa
import numpy as np
from scipy.signal import find_peaks

from config import AUDIO_CONFIG, AudioConfig
from .preprocessing import ProcessedAudio
from .schemas import FeatureSummary


def clamp(value: float) -> float:
    return round(float(np.clip(value, 0.0, 1.0)), 4)


def stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": round(float(np.mean(values)), 6),
        "std": round(float(np.std(values)), 6),
        "max": round(float(np.max(values)), 6),
    }


@dataclass(frozen=True)
class ExtractedFeatures:
    public: FeatureSummary
    signals: dict[str, float]


class AudioFeatureExtractor:
    def __init__(self, config: AudioConfig = AUDIO_CONFIG):
        self.config = config

    def extract(self, audio: ProcessedAudio) -> ExtractedFeatures:
        y, sr = audio.samples, audio.sample_rate
        stft = np.abs(librosa.stft(y, n_fft=self.config.n_fft, hop_length=self.config.hop_length))
        power = np.square(stft)
        total_power = np.sum(power, axis=0) + 1e-12
        frequencies = librosa.fft_frequencies(sr=sr, n_fft=self.config.n_fft)

        rms = librosa.feature.rms(S=stft)[0]
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.config.hop_length)[0]
        centroid = librosa.feature.spectral_centroid(S=stft, sr=sr)[0]
        bandwidth = librosa.feature.spectral_bandwidth(S=stft, sr=sr)[0]
        rolloff = librosa.feature.spectral_rolloff(S=stft, sr=sr, roll_percent=0.85)[0]
        flatness = librosa.feature.spectral_flatness(S=stft)[0]
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.config.n_mfcc,
                                   n_fft=self.config.n_fft, hop_length=self.config.hop_length)
        onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.config.hop_length)

        def band_ratio(bounds: tuple[float, float]) -> tuple[float, np.ndarray]:
            mask = (frequencies >= bounds[0]) & (frequencies < min(bounds[1], sr / 2))
            frame_ratios = np.sum(power[mask], axis=0) / total_power if np.any(mask) else np.zeros_like(total_power)
            return float(np.mean(frame_ratios)), frame_ratios

        low, low_frames = band_ratio(self.config.low_band_hz)
        mid, mid_frames = band_ratio(self.config.mid_band_hz)
        high, high_frames = band_ratio(self.config.high_band_hz)
        onset_norm = onset / (float(np.max(onset)) + 1e-12)
        min_peak_distance = max(1, int(0.08 * sr / self.config.hop_length))
        peaks, _ = find_peaks(onset_norm, height=0.35, distance=min_peak_distance)
        intervals = np.diff(peaks) * self.config.hop_length / sr
        irregularity = float(np.std(intervals) / (np.mean(intervals) + 1e-12)) if len(intervals) >= 2 else 0.0
        impulse_rate = len(peaks) / max(audio.duration_seconds, 1e-6)
        energy_cv = float(np.std(rms) / (np.mean(rms) + 1e-12))
        transient_ratio = float(np.percentile(onset, 95) / (np.mean(onset) + 1e-12))
        energy_variation_score = clamp(energy_cv / 1.2)
        impulse_activity = clamp(impulse_rate / 8.0) * energy_variation_score
        spectral_peak_concentration = float(np.mean(np.max(power, axis=0) / total_power))

        signals = {
            "transient_score": clamp((transient_ratio - 1.5) / 4.0),
            "low_mid_energy": clamp((low + mid) / 0.85),
            "impulse_rate_score": clamp(impulse_activity),
            "high_frequency_energy": clamp(high / 0.45),
            "spectral_flatness": clamp(float(np.mean(flatness)) / 0.5),
            "high_frequency_persistence": clamp(float(np.mean(high_frames > 0.20))),
            "onset_interval_irregularity": clamp(irregularity / 0.8),
            "temporal_energy_variation": energy_variation_score,
            "pulse_activity": clamp(len(peaks) / 8.0) * energy_variation_score,
            "tonal_concentration": clamp(spectral_peak_concentration / 0.45),
            "mid_frequency_energy": clamp(mid / 0.55),
            "energy_persistence": clamp(float(np.mean(rms > np.mean(rms) * 0.65))),
            "repetitive_component": clamp(1.0 - min(irregularity, 1.0)) if len(intervals) >= 2 else 0.0,
        }

        public = FeatureSummary(
            rms=stats(rms), zero_crossing_rate=stats(zcr), spectral_centroid=stats(centroid),
            spectral_bandwidth=stats(bandwidth), spectral_rolloff=stats(rolloff),
            mfcc_summary=[{"coefficient": index + 1, "mean": round(float(np.mean(row)), 6),
                           "std": round(float(np.std(row)), 6)} for index, row in enumerate(mfcc)],
            onset_strength=stats(onset), temporal_energy_variation=round(energy_cv, 6),
            frequency_band_energy={"low": round(low, 6), "mid": round(mid, 6), "high": round(high, 6)},
            measured_signals=signals,
        )
        return ExtractedFeatures(public=public, signals=signals)
