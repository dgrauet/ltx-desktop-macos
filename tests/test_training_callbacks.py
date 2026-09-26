"""Training subprocess callbacks: real per-step loss/lr (lib metrics_callback), samples via step_callback."""

from __future__ import annotations

from pathlib import Path

from engine.training import protocol
from engine.training.train_runner import make_training_callbacks


def test_metrics_callback_emits_real_step_lines() -> None:
    from ltx_trainer_mlx.trainer import StepMetrics

    lines: list[str] = []
    _, on_metrics = make_training_callbacks(lines.append)
    on_metrics(StepMetrics(step=3, total_steps=10, loss=0.0421, lr=2e-4, step_time_s=1.5, peak_memory_gb=12.5))

    evt = protocol.parse_line(lines[0])
    assert evt == {"type": "step", "step": 3, "loss": 0.0421, "lr": 2e-4, "peak_mem_gb": 12.5}


def test_metrics_without_peak_memory() -> None:
    from ltx_trainer_mlx.trainer import StepMetrics

    lines: list[str] = []
    _, on_metrics = make_training_callbacks(lines.append)
    on_metrics(StepMetrics(step=1, total_steps=2, loss=0.1, lr=1e-4, step_time_s=1.0, peak_memory_gb=None))
    assert protocol.parse_line(lines[0])["peak_mem_gb"] == 0.0


def test_step_callback_only_reports_samples() -> None:
    lines: list[str] = []
    on_step, _ = make_training_callbacks(lines.append)
    on_step(5, 10, [Path("/tmp/a.mp4"), Path("/tmp/b.mp4")])
    on_step(6, 10, [])
    assert lines == ["SAMPLE:/tmp/a.mp4", "SAMPLE:/tmp/b.mp4"]


def test_callbacks_bind_to_trainer_signature() -> None:
    import inspect

    from ltx_trainer_mlx.trainer import LtxvTrainer

    on_step, on_metrics = make_training_callbacks(lambda _: None)
    inspect.signature(LtxvTrainer.train).bind(
        None, disable_progress_bars=True, step_callback=on_step, metrics_callback=on_metrics,
    )
