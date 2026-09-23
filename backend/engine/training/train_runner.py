"""Subprocess: train a T2V LoRA via LtxvTrainer.

Progress AND result events go to stderr via the protocol (including DONE:, which
is a protocol event parsed by ``protocol.parse_line`` — NOT a stdout result).
stdout is left unused.

Emits protocol lines on stderr:
  STATUS:<msg>          — human-readable phase label
  STEP:<n>:<loss>:<lr>:<peak_gb>  — every optimizer step, from the lib's
                                    metrics_callback (loss averaged over
                                    gradient-accumulation micro-batches; lr used
                                    for that update; peak = MLX high-water mark)
  SAMPLE:<path>         — one line per sampled validation video
  DONE:<lora_path>      — final checkpoint path (full run only)
  PREFLIGHT_PEAK_GB:<v> — peak memory after N steps (--preflight mode)
  ERROR:<msg>           — fatal error

Modes::

  Full run:   --steps N  → train N steps, emits DONE:<path>
  Preflight:  --preflight N --steps N  → N steps + forced validation,
                                         emits PREFLIGHT_PEAK_GB:<v>, no final LoRA

Trainer hooks (ltx-trainer-mlx, see make_training_callbacks):
    step_callback(current_step, total_steps, sampled_video_paths) — after
        validation/checkpointing; used only for SAMPLE lines.
    metrics_callback(StepMetrics) — once per optimizer step; drives STEP lines.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _progress(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def make_training_callbacks(emit):
    """Build the trainer's ``(step_callback, metrics_callback)`` pair.

    Args:
        emit: Sink for protocol lines (stderr in the subprocess).
    """
    from engine.training import protocol  # noqa: PLC0415

    def step_callback(current_step: int, total_steps: int, sampled_video_paths: list[Path]) -> None:
        for video_path in sampled_video_paths:
            emit(protocol.format_sample(str(video_path)))

    def metrics_callback(metrics) -> None:
        emit(protocol.format_step(
            step=metrics.step,
            loss=float(metrics.loss),
            lr=float(metrics.lr),
            peak_gb=float(metrics.peak_memory_gb or 0.0),
        ))

    return step_callback, metrics_callback


def main() -> int:  # noqa: PLR0911  (multiple return paths are intentional)
    ap = argparse.ArgumentParser(
        description="Train a T2V LoRA via LtxvTrainer (single subprocess, stderr progress).",
    )
    ap.add_argument(
        "--data-root",
        required=True,
        metavar="DIR",
        help="Root of the preprocessed dataset (.precomputed/ structure).",
    )
    ap.add_argument(
        "--model",
        required=True,
        metavar="PATH",
        help="Path to the transformer weights directory.",
    )
    ap.add_argument(
        "--text-encoder",
        required=True,
        metavar="PATH_OR_ID",
        help="Path or HuggingFace ID for the Gemma 3 12B text encoder.",
    )
    ap.add_argument(
        "--output",
        required=True,
        metavar="DIR",
        help="Directory for checkpoints and logs.",
    )
    ap.add_argument(
        "--steps",
        type=int,
        required=True,
        metavar="N",
        help="Number of optimisation steps.",
    )
    ap.add_argument(
        "--rank",
        type=int,
        default=32,
        metavar="N",
        help="LoRA rank. Default: 32.",
    )
    ap.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
        metavar="LR",
        help="AdamW learning rate. Default: 5e-4.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="N",
        help="Global RNG seed. Default: 42.",
    )
    ap.add_argument(
        "--preflight",
        type=int,
        default=0,
        metavar="N",
        help=(
            "If >0, run N steps to measure peak memory then exit without "
            "producing a final LoRA. Emits PREFLIGHT_PEAK_GB:<value>."
        ),
    )
    ap.add_argument(
        "--low-ram",
        action="store_true",
        help=(
            "Enable 32GB-safe overrides: batch_size=1, gradient checkpointing. "
            "Recommended for quantized models or machines with ≤32GB RAM."
        ),
    )
    ap.add_argument(
        "--validate",
        action="store_true",
        help=(
            "Generate periodic validation preview samples. Off by default: the "
            "step-0 validation is a sustained-Metal inference that trips the macOS "
            "GPU watchdog (SIGKILL) and raises peak memory. Samples are unused by "
            "the UI, so leave off unless you specifically want previews."
        ),
    )
    args = ap.parse_args()

    # Heavy imports kept inside main() so --help exits fast.
    import os  # noqa: PLC0415

    import mlx.core as mx  # noqa: PLC0415

    # Cap the MLX buffer-reuse cache. MLX hoards freed Metal buffers for reuse;
    # during training the cache can balloon (~18GB observed on 32GB), pushing the
    # memory compressor to exhaustion and triggering a jetsam OOM SIGKILL on the
    # first step. A small cache forces buffers back to the OS. Env-tunable;
    # default 1GB. Set LTX_MLX_CACHE_LIMIT_GB=0 to disable the cap.
    _cache_gb = float(os.environ.get("LTX_MLX_CACHE_LIMIT_GB", "1"))
    if _cache_gb > 0:
        mx.set_cache_limit(int(_cache_gb * 1024**3))

    from ltx_trainer_mlx.trainer import LtxvTrainer  # noqa: PLC0415

    from engine.training import protocol  # noqa: PLC0415
    from engine.training.config_builder import build_t2v_config  # noqa: PLC0415

    steps = args.preflight if args.preflight > 0 else args.steps
    cfg = build_t2v_config(
        model_path=args.model,
        text_encoder_path=args.text_encoder,
        preprocessed_data_root=args.data_root,
        output_dir=args.output,
        steps=steps,
        rank=args.rank,
        learning_rate=args.learning_rate,
        seed=args.seed,
        video_dims=(704, 480, 25),
        low_ram=args.low_ram,
        enable_validation=args.validate,
    )

    step_callback, metrics_callback = make_training_callbacks(_progress)

    _progress("STATUS:Loading model")
    trainer = LtxvTrainer(cfg)

    _progress("STATUS:Training")
    try:
        ckpt_path, train_stats = trainer.train(
            disable_progress_bars=True,
            step_callback=step_callback,
            metrics_callback=metrics_callback,
        )
    except Exception as exc:  # noqa: BLE001 — surface OOM/errors to parent
        _progress(protocol.format_error(f"{type(exc).__name__}: {exc}"))
        return 1

    peak_gb = train_stats.peak_memory_gb

    if args.preflight > 0:
        _progress(f"PREFLIGHT_PEAK_GB:{peak_gb:.2f}")
        return 0

    _progress("STATUS:Done")
    _progress(protocol.format_done(str(ckpt_path)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
