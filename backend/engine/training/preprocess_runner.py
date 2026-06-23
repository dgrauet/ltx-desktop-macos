"""Subprocess: preprocess a training dataset into cached latents/embeddings.

Emits STATUS: lines on stderr (mlx_runner convention). stdout is reserved.

Usage::

    python -m engine.training.preprocess_runner \\
        --manifest <videos_dir> \\
        --out <output_dir> \\
        --model <model_dir>

``--manifest`` is the directory of raw video files (named for symmetry with the
training API; maps to ``preprocess_dataset``'s ``videos_dir`` argument).

DISCOVERY NOTE — real preprocessing API (ltx_trainer_mlx.preprocess):
    preprocess_dataset(
        videos_dir: str,       # directory of raw video files
        output_dir: str,       # where to write .precomputed/
        model_dir: str,        # single model dir (VAE + text encoder bundled)
        gemma_model_id: str = "mlx-community/gemma-3-12b-it-4bit",
        target_height: int | None = None,
        target_width: int | None = None,
        max_frames: int = 97,
        captions_dir: str | None = None,
        caption_ext: str = ".txt",
    ) -> None

The lib takes a single ``model_dir`` (not separate model + text_encoder paths).
``--manifest`` maps to ``videos_dir``; ``--model`` maps to ``model_dir``.
``--text-encoder`` is accepted but ignored (the lib bundles encoder in model_dir).
"""
from __future__ import annotations

import argparse
import sys


def _progress(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Preprocess a video dataset into training-ready latents and embeddings.",
    )
    ap.add_argument(
        "--manifest",
        required=True,
        metavar="VIDEOS_DIR",
        help="Directory containing raw video files to preprocess.",
    )
    ap.add_argument(
        "--out",
        required=True,
        metavar="OUTPUT_DIR",
        help="Output directory for preprocessed data (.precomputed/ structure).",
    )
    ap.add_argument(
        "--model",
        required=True,
        metavar="MODEL_DIR",
        help="Model directory containing VAE encoder weights (and Gemma connector).",
    )
    ap.add_argument(
        "--text-encoder",
        default=None,
        metavar="TEXT_ENCODER",
        help=(
            "Accepted for API symmetry but ignored: the lib bundles the text encoder "
            "inside --model. Pass a HuggingFace ID here to override gemma_model_id."
        ),
    )
    ap.add_argument(
        "--captions-dir",
        default=None,
        metavar="CAPTIONS_DIR",
        help="Directory with .txt caption files (one per video, same stem). "
             "Omit to use video filenames as captions.",
    )
    ap.add_argument(
        "--max-frames",
        type=int,
        default=97,
        metavar="N",
        help="Maximum frames per video (must satisfy N %% 8 == 1). Default: 97.",
    )
    args = ap.parse_args()

    _progress("STATUS:Preprocessing dataset")

    # Heavy import stays inside main() so --help exits fast.
    from ltx_trainer_mlx.preprocess import preprocess_dataset  # noqa: PLC0415

    gemma_model_id = args.text_encoder or "mlx-community/gemma-3-12b-it-4bit"

    preprocess_dataset(
        videos_dir=args.manifest,
        output_dir=args.out,
        model_dir=args.model,
        gemma_model_id=gemma_model_id,
        captions_dir=args.captions_dir,
        max_frames=args.max_frames,
    )

    _progress("STATUS:Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
