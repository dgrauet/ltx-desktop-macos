"""ETA countdown built from the lib's ``[estimate] … ~X remaining`` stderr lines."""

from __future__ import annotations

import pytest

from engine.mlx_runner import EtaTracker, parse_duration, preview_enabled


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("45 s", 45), ("1 min 30 s", 90), ("2 h 5 min", 7500), ("nonsense", None)],
)
def test_parse_duration(text: str, seconds: float | None) -> None:
    assert parse_duration(text) == seconds


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def test_countdown_follows_projection() -> None:
    clock = _Clock()
    eta = EtaTracker(clock)
    assert eta.status("Generating video") == "Generating video"

    # The work announcement line is consumed but carries no projection.
    assert eta.feed("[estimate] denoising: 8 steps x 1 passes over 384 video tokens = 8 forwards")
    assert eta.status("Generating video") == "Generating video"

    assert eta.feed("[estimate] denoising: ~1 min 30 s remaining (6.4 s/forward) (refined)")
    clock.now += 20
    assert eta.status("Generating video") == "Generating video — ~1m 10s left in this pass"
    clock.now += 60
    assert eta.status("Generating video") == "Generating video — ~10s left in this pass"
    clock.now += 30
    assert eta.status("Generating video") == "Generating video"


def test_reset_and_non_estimate_lines() -> None:
    eta = EtaTracker(_Clock())
    assert not eta.feed("STATUS:Denoising stage 2 (3 steps)")
    eta.feed("[estimate] denoising: ~36 s remaining (18.0 s/forward)")
    eta.reset()
    assert eta.status("x") == "x"


def test_preview_disabled_in_low_ram_and_by_env(monkeypatch) -> None:
    monkeypatch.delenv("LTX_PREVIEW", raising=False)
    assert preview_enabled(low_ram=False)
    assert not preview_enabled(low_ram=True)
    monkeypatch.setenv("LTX_PREVIEW", "0")
    assert not preview_enabled(low_ram=False)
