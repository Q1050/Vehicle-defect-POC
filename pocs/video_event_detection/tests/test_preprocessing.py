import pytest

from video_diagnostics.preprocessing import VideoFrameSampler, VideoValidationError


def test_metadata_and_sampling(generated_clips):
    path = generated_clips / "synthetic_static.mp4"
    sampled = VideoFrameSampler().sample(path.read_bytes(), path.name)
    assert sampled.metadata.width == 320
    assert sampled.metadata.height == 240
    assert sampled.metadata.fps == pytest.approx(15, abs=0.2)
    assert sampled.metadata.frame_count == 45
    assert 10 <= sampled.metadata.frames_analyzed < 45
    assert len(sampled.metadata.timestamps_seconds) == sampled.metadata.frames_analyzed


@pytest.mark.parametrize("payload, filename", [(b"", "empty.mp4"), (b"not video", "bad.mp4")])
def test_invalid_media_rejected(payload, filename):
    with pytest.raises(VideoValidationError):
        VideoFrameSampler().sample(payload, filename)


def test_unsupported_extension_rejected():
    with pytest.raises(VideoValidationError, match="Unsupported"):
        VideoFrameSampler().sample(b"data", "video.txt")
