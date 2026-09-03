"""Generate synthetic demo signals. They do not represent real vehicle faults."""

from pathlib import Path

import numpy as np
import soundfile as sf


SAMPLE_RATE = 22_050
DURATION = 4.0
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "samples"


def signals() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    t = np.arange(int(SAMPLE_RATE * DURATION)) / SAMPLE_RATE
    clean = 0.3 * np.sin(2 * np.pi * 180 * t)

    impulses = 0.03 * np.sin(2 * np.pi * 150 * t)
    for position in np.arange(0.25, DURATION, 0.3):
        start = int(position * SAMPLE_RATE)
        length = min(180, len(t) - start)
        impulses[start:start + length] += 0.9 * np.hanning(length)

    hiss = 0.03 * np.sin(2 * np.pi * 160 * t)
    high_noise = rng.normal(0, 0.22, len(t))
    spectrum = np.fft.rfft(high_noise)
    spectrum[np.fft.rfftfreq(len(t), 1 / SAMPLE_RATE) < 3_000] = 0
    hiss += np.fft.irfft(spectrum, len(t)).astype(np.float32)

    irregular = 0.02 * np.sin(2 * np.pi * 140 * t)
    for position in (0.2, 0.48, 1.1, 1.38, 2.25, 2.48, 3.45):
        start = int(position * SAMPLE_RATE)
        length = min(250, len(t) - start)
        irregular[start:start + length] += 0.8 * np.hanning(length)

    return {
        "synthetic_clean.wav": clean,
        "synthetic_impulses.wav": impulses,
        "synthetic_hiss.wav": hiss,
        "synthetic_irregular.wav": irregular,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, audio in signals().items():
        sf.write(OUTPUT_DIR / filename, np.clip(audio, -1, 1), SAMPLE_RATE)
        print(f"Wrote {OUTPUT_DIR / filename}")


if __name__ == "__main__":
    main()
