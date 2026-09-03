"""Generate tiny synthetic clips for software tests, not vehicle-fault simulation."""

from pathlib import Path

import cv2
import numpy as np


OUTPUT = Path(__file__).resolve().parents[1] / "samples"
FPS, SIZE, FRAME_COUNT = 15, (320, 240), 45


def base_frame() -> np.ndarray:
    frame = np.full((SIZE[1], SIZE[0], 3), (45, 70, 95), dtype=np.uint8)
    cv2.line(frame, (0, 40), (319, 180), (180, 80, 40), 5)
    cv2.circle(frame, (75, 150), 35, (30, 170, 100), -1)
    cv2.putText(frame, "POC", (190, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (220, 120, 30), 2)
    return frame


def clips() -> dict[str, list[np.ndarray]]:
    base = base_frame()
    static = [base.copy() for _ in range(FRAME_COUNT)]
    global_motion = []
    vibration = []
    smoke = []
    for index in range(FRAME_COUNT):
        shift = int(round(5 * np.sin(index * 0.45)))
        matrix = np.float32([[1, 0, shift], [0, 1, 2]])
        global_motion.append(cv2.warpAffine(base, matrix, SIZE, borderMode=cv2.BORDER_REFLECT))

        local = base.copy()
        offset = int(round(15 * np.sin(index * 0.9)))
        cv2.rectangle(local, (125 + offset, 120), (205 + offset, 190), (20, 210, 230), -1)
        vibration.append(local)

        plume = base.copy()
        center = (215 + int(index * 0.4), 190 - int(index * 1.5))
        axes = (18 + index // 5, 28 + index // 4)
        cv2.ellipse(plume, center, axes, 0, 0, 360, (185, 185, 185), -1)
        plume = cv2.GaussianBlur(plume, (9, 9), 0)
        smoke.append(plume)
    return {
        "synthetic_static.mp4": static,
        "synthetic_global_motion.mp4": global_motion,
        "synthetic_vibration.mp4": vibration,
        "synthetic_smoke_like.mp4": smoke,
    }


def write_clip(path: Path, frames: list[np.ndarray]) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create an MP4 fixture.")
    for frame in frames:
        writer.write(frame)
    writer.release()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for filename, frames in clips().items():
        path = OUTPUT / filename
        write_clip(path, frames)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
