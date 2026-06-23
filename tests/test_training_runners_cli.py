"""P0: runner scripts expose the expected CLI without importing heavy weights."""
from __future__ import annotations

import subprocess
import sys


def _help(mod: str) -> tuple[int, str]:
    r = subprocess.run(
        [sys.executable, "-m", mod, "--help"],
        capture_output=True,
        text=True,
        cwd="backend",
    )
    return r.returncode, r.stdout + r.stderr


def test_preprocess_runner_help() -> None:
    code, out = _help("engine.training.preprocess_runner")
    assert code == 0
    assert "--manifest" in out and "--out" in out


def test_train_runner_help() -> None:
    code, out = _help("engine.training.train_runner")
    assert code == 0
    assert "--preflight" in out and "--steps" in out
