"""Unit test for probe_frame_count — used to reject too-short IC-LoRA control videos.

IC-LoRA's reference-video VAE encoder requires a (1 + 8k)-frame input and the
library forces a minimum of 9 frames, so a control video with fewer than 9
frames crashes deep in the VAE with a cryptic reshape error. We probe the
frame count up front to reject it with a clear message.
"""

import subprocess
from pathlib import Path

from engine.ffmpeg_utils import probe_frame_count


def _make(path: Path, frames: int) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i",
         f"testsrc=size=128x128:rate=24", "-frames:v", str(frames),
         "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True,
    )


def test_probe_frame_count_short(tmp_path):
    v = tmp_path / "short.mp4"
    _make(v, 7)
    assert probe_frame_count(str(v)) == 7


def test_probe_frame_count_long(tmp_path):
    v = tmp_path / "long.mp4"
    _make(v, 25)
    assert probe_frame_count(str(v)) == 25
