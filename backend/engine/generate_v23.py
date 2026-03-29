"""LTX-2.3 generation subprocess -- delegates to ltx-pipelines-mlx.

Invoked as: python -m engine.generate_v23 --mode t2v --prompt "..." --output-path out.mp4 ...
Emits progress on stderr in the format parsed by mlx_runner.py:
  STATUS:<message>
  MEMORY:<label>:active=<gb>:cache=<gb>:peak=<gb>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx

from engine.memory_manager import aggressive_cleanup, get_memory_stats


# ---------------------------------------------------------------------------
# Progress helpers (stderr protocol for mlx_runner.py)
# ---------------------------------------------------------------------------

def _progress(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _report_memory(label: str) -> None:
    stats = get_memory_stats()
    _progress(
        f"MEMORY:{label}"
        f":active={stats['active_memory_gb']:.3f}"
        f":cache={stats['cache_memory_gb']:.3f}"
        f":peak={stats['peak_memory_gb']:.3f}"
    )


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

def _create_pipeline(args: argparse.Namespace):
    """Instantiate the correct library pipeline for the given mode."""
    from ltx_pipelines_mlx import (
        ExtendPipeline,
        ImageToVideoPipeline,
        RetakePipeline,
        TextToVideoPipeline,
    )

    model_dir = args.model_dir
    gemma = args.gemma or "mlx-community/gemma-3-12b-it-4bit"
    low_memory = True

    if args.mode == "retake":
        return RetakePipeline(model_dir, gemma_model_id=gemma, low_memory=low_memory)
    elif args.mode == "extend":
        return ExtendPipeline(model_dir, gemma_model_id=gemma, low_memory=low_memory)
    elif args.mode == "i2v":
        return ImageToVideoPipeline(model_dir, gemma_model_id=gemma, low_memory=low_memory)
    else:  # t2v (default)
        return TextToVideoPipeline(model_dir, gemma_model_id=gemma, low_memory=low_memory)


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

def _run_t2v(pipeline, args: argparse.Namespace) -> None:
    """Text-to-video or image-to-video generation."""
    _progress("STATUS:Loading model")
    _report_memory("before_load")

    pipeline.load()

    _report_memory("after_model_load")
    _progress("STATUS:Generating video")

    # Build kwargs matching the library's generate_and_save signature
    gen_kwargs: dict = {
        "prompt": args.prompt,
        "output_path": args.output_path,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "seed": args.seed,
        "num_steps": args.num_steps,
    }

    # I2V-specific: pass image if the pipeline supports it
    if args.mode == "i2v" and args.image:
        import inspect
        i2v_sig = inspect.signature(pipeline.generate_and_save)
        if "image" in i2v_sig.parameters:
            gen_kwargs["image"] = args.image

    pipeline.generate_and_save(**gen_kwargs)

    _report_memory("after_generation")
    _progress("STATUS:Done")


def _run_retake(pipeline, args: argparse.Namespace) -> None:
    """Retake: regenerate a frame range in an existing video."""
    _progress("STATUS:Loading model")
    _report_memory("before_load")

    pipeline.load()

    _report_memory("after_model_load")
    _progress("STATUS:Retaking segment")

    video_latent, audio_latent = pipeline.retake_from_video(
        prompt=args.prompt,
        video_path=args.retake_source,
        start_frame=args.retake_start_frame,
        end_frame=args.retake_end_frame,
        seed=args.seed,
        num_steps=args.num_steps or 30,
    )

    _progress("STATUS:Decoding video")
    pipeline._decode_and_save_video(video_latent, audio_latent, args.output_path)

    _report_memory("after_generation")
    _progress("STATUS:Done")


def _run_extend(pipeline, args: argparse.Namespace) -> None:
    """Extend: add frames before or after an existing video."""
    _progress("STATUS:Loading model")
    _report_memory("before_load")

    pipeline.load()

    _report_memory("after_model_load")
    _progress("STATUS:Extending video")

    video_latent, audio_latent = pipeline.extend_from_video(
        prompt=args.prompt,
        video_path=args.extend_source,
        extend_frames=args.extend_frames,
        direction=args.extend_direction,
        seed=args.seed,
        num_steps=args.num_steps or 30,
    )

    _progress("STATUS:Decoding video")
    pipeline._decode_and_save_video(video_latent, audio_latent, args.output_path)

    _report_memory("after_generation")
    _progress("STATUS:Done")


# ---------------------------------------------------------------------------
# LoRA arg parsing
# ---------------------------------------------------------------------------

def _parse_lora_args(lora_list: list[str]) -> list[tuple[str, float]]:
    """Parse --lora path:strength args into list of (path, strength) tuples."""
    result = []
    for entry in lora_list:
        idx = entry.rfind(":")
        if idx > 0 and idx < len(entry) - 1:
            try:
                strength = float(entry[idx + 1:])
                path = entry[:idx]
                result.append((path, strength))
                continue
            except ValueError:
                pass
        result.append((entry, 0.7))
    return result


# ---------------------------------------------------------------------------
# Prompt enhancement (standalone subprocess mode)
# ---------------------------------------------------------------------------

def _run_enhance(args: argparse.Namespace) -> None:
    """Enhance a prompt using Gemma via the library text encoder."""
    from ltx_core_mlx.text_encoders.gemma.encoders.base_encoder import GemmaLanguageModel

    _progress("STATUS:Loading Gemma for enhancement")
    gemma = GemmaLanguageModel(args.gemma or "mlx-community/gemma-3-12b-it-4bit")
    gemma.load()

    if args.enhance_mode == "i2v":
        enhanced = gemma.enhance_i2v(args.prompt, seed=args.seed)
    else:
        enhanced = gemma.enhance_t2v(args.prompt, seed=args.seed)

    # Write enhanced prompt to stdout (not stderr -- stderr is for progress)
    print(enhanced, flush=True)

    del gemma
    aggressive_cleanup()
    _progress("STATUS:Done")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LTX-2.3 generation subprocess")

    parser.add_argument("--mode", choices=["t2v", "i2v", "retake", "extend", "enhance"],
                        default="t2v", help="Pipeline mode")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model-dir", required=True, help="HF model path or repo ID")
    parser.add_argument("--output-path", default="output.mp4")
    parser.add_argument("--gemma", default=None, help="Gemma model ID for text encoding")

    # Video params
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num-frames", type=int, default=97)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--num-steps", type=int, default=8)

    # I2V
    parser.add_argument("--image", default=None, help="Reference image path for I2V")
    parser.add_argument("--image-strength", type=float, default=1.0)

    # Retake
    parser.add_argument("--retake-source", default=None, help="Source video for retake")
    parser.add_argument("--retake-start-frame", type=int, default=0)
    parser.add_argument("--retake-end-frame", type=int, default=-1)

    # Extend
    parser.add_argument("--extend-source", default=None, help="Source video for extend")
    parser.add_argument("--extend-frames", type=int, default=49)
    parser.add_argument("--extend-direction", choices=["before", "after"], default="after")

    # LoRA
    parser.add_argument("--lora", action="append", default=None,
                        help="LoRA path:strength (can repeat)")

    # Enhancement
    parser.add_argument("--enhance-prompt", action="store_true",
                        help="Enhance prompt via Gemma before generation")
    parser.add_argument("--enhance-mode", choices=["t2v", "i2v"], default="t2v")

    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.mode == "enhance":
        _run_enhance(args)
        return

    pipeline = _create_pipeline(args)

    if args.mode == "retake":
        _run_retake(pipeline, args)
    elif args.mode == "extend":
        _run_extend(pipeline, args)
    else:
        _run_t2v(pipeline, args)

    # Final cleanup
    del pipeline
    aggressive_cleanup()


if __name__ == "__main__":
    main()
