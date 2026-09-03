"""Stream an existing synthetic WAV over the realtime WebSocket protocol."""

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
import websockets


DEFAULT_AUDIO = (
    Path(__file__).resolve().parents[2]
    / "audio_diagnostics"
    / "samples"
    / "synthetic_impulses.wav"
)


def load_pcm(path: Path, target_rate: int = 16_000) -> np.ndarray:
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = np.mean(samples, axis=1)
    if sample_rate != target_rate:
        divisor = int(np.gcd(sample_rate, target_rate))
        mono = resample_poly(mono, target_rate // divisor, sample_rate // divisor)
    return np.round(np.clip(mono, -1, 1) * 32767).astype("<i2")


async def stream(path: Path, uri: str, realtime: bool) -> None:
    pcm = load_pcm(path)
    chunk_samples = 4_000  # 250 ms at 16 kHz
    update_count = 0
    async with websockets.connect(uri, max_size=2**20) as websocket:
        await websocket.send(json.dumps({
            "type": "start",
            "sample_rate": 16_000,
            "channels": 1,
            "sample_format": "pcm_s16le",
            "user_description": "Synthetic transport test; not a real vehicle diagnosis.",
        }))
        print(await websocket.recv())

        for offset in range(0, len(pcm), chunk_samples):
            await websocket.send(pcm[offset:offset + chunk_samples].tobytes())
            if realtime:
                await asyncio.sleep(chunk_samples / 16_000)

        await websocket.send(json.dumps({"type": "stop"}))
        async for response in websocket:
            print(response)
            payload = json.loads(response)
            if payload.get("type") == "diagnostic_update":
                update_count += 1
            if payload.get("type") == "session_ended":
                break
    if update_count < 1:
        raise RuntimeError("No diagnostic window was returned.")
    print(f"Received {update_count} diagnostic update(s); session ended cleanly.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--uri", default="ws://127.0.0.1:8000/ws/audio")
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Pace chunks in realtime; default fast mode sends without sleeping.",
    )
    args = parser.parse_args()
    asyncio.run(stream(args.audio, args.uri, args.realtime))


if __name__ == "__main__":
    main()
