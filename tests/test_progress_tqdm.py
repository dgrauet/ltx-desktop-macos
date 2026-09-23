"""The tqdm hook must number denoise loops even when a sampler breaks out early.

The 2.5 ancestral sampler ``break``s on its last step, so code after the
generator's ``for`` never runs; stage 2 was reported as stage 1 again.
"""

from __future__ import annotations

import engine.generate_v23 as g


def _run_loop(desc: str, n: int, *, break_last: bool) -> None:
    for i, _ in enumerate(g._ProgressTqdm(range(n), desc=desc)):
        if break_last and i == n - 1:
            break


def test_stage_advances_after_early_break(monkeypatch, capsys) -> None:
    monkeypatch.setattr(g, "_current_stage", 1)
    monkeypatch.setattr(g, "_denoise_loop_started", False)

    _run_loop("Denoising (ancestral)", 8, break_last=True)
    _run_loop("Denoising", 3, break_last=False)

    err = capsys.readouterr().err
    assert "STATUS:Denoising stage 1 (8 steps)" in err
    assert "STATUS:Denoising stage 2 (3 steps)" in err
    assert "STAGE:2:STEP:3:3" in err


def test_non_denoise_bars_do_not_advance_stage(monkeypatch, capsys) -> None:
    monkeypatch.setattr(g, "_current_stage", 1)
    monkeypatch.setattr(g, "_denoise_loop_started", False)

    _run_loop("Denoising", 2, break_last=False)
    _run_loop("Decoding", 2, break_last=False)

    assert g._current_stage == 1
