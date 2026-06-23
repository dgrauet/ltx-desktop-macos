"""Subprocess: train a T2V LoRA via LtxvTrainer; stderr progress; stdout reserved.

Emits protocol lines on stderr:
  STATUS:<msg>          — human-readable phase label
  STEP:<n>:<loss>:<lr>:<peak_gb>  — sampled at validation steps (loss/lr are 0.0
                                    placeholders; the lib's StepCallback provides no
                                    per-step loss in P0)
  SAMPLE:<path>         — one line per sampled validation video
  DONE:<lora_path>      — final checkpoint path (full run only)
  PREFLIGHT_PEAK_GB:<v> — peak memory after N steps (--preflight mode)
  ERROR:<msg>           — fatal error

Modes::

  Full run:   --steps N  → train N steps, emits DONE:<path>
  Preflight:  --preflight N --steps N  → N steps + forced validation,
                                         emits PREFLIGHT_PEAK_GB:<v>, no final LoRA

StepCallback contract (ltx_trainer_mlx 0.14):
    StepCallback = Callable[[int, int, list[Path]], None]
    args: (current_step, total_steps, sampled_video_paths)
Called ONLY when validation videos are sampled (NOT every step).
No loss or lr values are available from the callback — placeholders 0.0 are emitted.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _progress(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


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
        "--preflight",
        type=int,
        default=0,
        metavar="N",
        help=(
            "If >0, run N steps to measure peak memory then exit without "
            "producing a final LoRA. Emits PREFLIGHT_PEAK_GB:<value>."
        ),
    )
    args = ap.parse_args()

    # Heavy imports kept inside main() so --help exits fast.
    import mlx.core as mx  # noqa: PLC0415
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
        video_dims=(704, 480, 25),
    )

    def step_callback(current_step: int, total_steps: int, sampled_video_paths: list[Path]) -> None:
        """Called by trainer when validation videos are sampled.

        The lib provides no per-step loss or lr in P0; placeholders 0.0 are
        emitted so the protocol line is structurally valid for later parsing.
        """
        peak_gb = mx.get_peak_memory() / (1024 ** 3)
        _progress(protocol.format_step(step=current_step, loss=0.0, lr=0.0, peak_gb=peak_gb))
        for video_path in sampled_video_paths:
            _progress(protocol.format_sample(str(video_path)))

    _progress("STATUS:Loading model")
    trainer = LtxvTrainer(cfg)

    _progress("STATUS:Training")
    try:
        ckpt_path, train_stats = trainer.train(
            disable_progress_bars=True,
            step_callback=step_callback,
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
