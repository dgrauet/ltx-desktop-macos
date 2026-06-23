"""P0: dataset_store validates media constraints + caption adequacy."""
import subprocess
from pathlib import Path

import pytest

from dataset_store import build_manifest, validate_clip, MIN_CLIPS
from engine.ffmpeg_utils import find_ffmpeg


def _make_clip(path: Path, frames: int, w: int, h: int):
    # frames at 25fps; testsrc gives exact frame count via -frames:v
    subprocess.run(
        [find_ffmpeg(), "-y", "-f", "lavfi", "-i",
         f"testsrc=size={w}x{h}:rate=25", "-frames:v", str(frames),
         "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True,
    )


def test_valid_clip_has_no_violations(tmp_path):
    clip = tmp_path / "ok.mp4"
    _make_clip(clip, frames=25, w=704, h=480)  # 25 % 8 == 1, dims % 32 == 0
    assert validate_clip(str(clip)) == []


def test_bad_frame_count_flagged(tmp_path):
    clip = tmp_path / "bad.mp4"
    _make_clip(clip, frames=24, w=704, h=480)  # 24 % 8 != 1
    violations = validate_clip(str(clip))
    assert any("frame" in v.lower() for v in violations)


def test_empty_caption_rejected():
    with pytest.raises(ValueError):
        build_manifest([{"caption": "", "video": "a.mp4"}])


def test_few_clips_warns_not_raises():
    m = build_manifest([{"caption": "a cat", "video": "a.mp4"}])
    assert m[0]["caption"] == "a cat"
    # adequacy warning surfaced separately, not an exception
    assert MIN_CLIPS == 5
