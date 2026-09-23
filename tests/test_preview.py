"""Progressive preview: each written WebP is announced exactly once as PREVIEW:<path>."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from engine.preview import EmittingPreview, make_preview


def _preview(tmp_path: Path, emitted: list[str]) -> EmittingPreview:
    preview = make_preview(str(tmp_path / "previews"), fps=24, seed=7, emit=emitted.append)
    # Skip the real VAE decode: return one tiny frame.
    preview._decode_window = lambda *a, **k: [Image.new("RGB", (8, 8), "red")]
    return preview


def _bind(preview: EmittingPreview, stage: int | None = None):
    return preview.bind(
        latent_frames=4, latent_height=2, latent_width=2,
        decoder_block=None, patchifier=None, stage=stage,
    )


def test_emits_each_preview_once_at_interval(tmp_path: Path) -> None:
    emitted: list[str] = []
    on_step = _bind(_preview(tmp_path, emitted))

    for step in range(5):  # interval 2 -> steps 0, 2, and the last (4)
        on_step(step, 5, None, 1.0)

    assert len(emitted) == 3
    assert all(line.startswith("PREVIEW:") and line.endswith(".webp") for line in emitted)
    assert len(set(emitted)) == 3
    for line in emitted:
        assert Path(line.removeprefix("PREVIEW:")).is_file()


def test_consumed_files_are_not_reannounced(tmp_path: Path) -> None:
    """mlx_runner deletes each file after reading; later steps must not re-emit it."""
    emitted: list[str] = []
    on_step = _bind(_preview(tmp_path, emitted))

    on_step(0, 3, None, 1.0)
    Path(emitted[0].removeprefix("PREVIEW:")).unlink()
    on_step(2, 3, None, 1.0)

    assert len(emitted) == 2


def test_two_stage_previews_are_distinct(tmp_path: Path) -> None:
    emitted: list[str] = []
    preview = _preview(tmp_path, emitted)
    _bind(preview, stage=1)(0, 2, None, 1.0)
    _bind(preview, stage=2)(0, 2, None, 1.0)
    assert len(emitted) == 2 and "_s1_" in emitted[0] and "_s2_" in emitted[1]


def test_decode_failure_disables_without_raising(tmp_path: Path) -> None:
    emitted: list[str] = []
    preview = _preview(tmp_path, emitted)

    def boom(*a, **k):
        raise RuntimeError("decoder OOM")

    preview._decode_window = boom
    on_step = _bind(preview)
    on_step(0, 3, None, 1.0)  # must not raise
    assert emitted == []
    assert _bind(preview) is None
