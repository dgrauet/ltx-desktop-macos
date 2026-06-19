"""LTX-2.3 generation subprocess -- delegates to ltx-pipelines-mlx.

Invoked as: python -m engine.generate_v23 --mode t2v --prompt "..." --output-path out.mp4 ...
Emits progress on stderr in the format parsed by mlx_runner.py:
  STATUS:<message>
  STAGE:<n>:STEP:<step>:<total>
  MEMORY:<label>:active=<gb>:cache=<gb>:peak=<gb>
"""

from __future__ import annotations

import argparse
import sys

from engine.memory_manager import aggressive_cleanup, get_memory_stats

# ---------------------------------------------------------------------------
# tqdm monkey-patch — intercept library progress bars to emit STAGE/STEP
# ---------------------------------------------------------------------------

_current_stage = 1


class _ProgressTqdm:
    """Drop-in tqdm replacement that emits STAGE:STEP lines on stderr."""

    def __init__(self, iterable=None, *, desc="", total=None, disable=False, **kwargs):
        global _current_stage
        self._iterable = iterable
        self._items = list(iterable) if iterable is not None else []
        self._total = total or len(self._items)
        self._desc = desc or ""
        self._disable = disable
        self._step = 0
        # Emit a STATUS for each new denoising loop
        if not disable and "denois" in self._desc.lower():
            _progress(f"STATUS:Denoising stage {_current_stage} ({self._total} steps)")

    def __iter__(self):
        global _current_stage
        for item in self._items:
            self._step += 1
            if not self._disable:
                _progress(f"STAGE:{_current_stage}:STEP:{self._step}:{self._total}")
            yield item
        # After completing a denoising loop, advance stage for next loop
        if not self._disable and "denois" in self._desc.lower():
            _current_stage += 1

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def update(self, n=1):
        self._step += n
        if not self._disable:
            _progress(f"STAGE:{_current_stage}:STEP:{self._step}:{self._total}")

    def close(self):
        pass

    def set_description(self, desc):
        self._desc = desc


def _install_tqdm_hook():
    """Replace tqdm in the library modules with our progress emitter."""
    import ltx_pipelines_mlx.utils.samplers as samplers_mod
    samplers_mod.tqdm = _ProgressTqdm
    try:
        import ltx_core_mlx.model.video_vae.video_vae as vae_mod
        if hasattr(vae_mod, "tqdm"):
            vae_mod.tqdm = _ProgressTqdm
    except (ImportError, AttributeError):
        pass




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
    """Instantiate the correct library pipeline for the given mode and pipeline type.

    Since ltx-2-mlx 0.10, I2V is supported on every pipeline via the
    ``image=`` kwarg (no dedicated ImageToVideoPipeline), and extend is
    folded into RetakePipeline.
    """
    from ltx_pipelines_mlx import (
        A2VidPipelineTwoStage,
        DistilledPipeline,
        RetakePipeline,
        TI2VidOneStagePipeline,
        TI2VidTwoStagesHQPipeline,
        TI2VidTwoStagesPipeline,
    )

    model_dir = args.model_dir
    common = {
        "gemma_model_id": args.gemma or "mlx-community/gemma-3-12b-it-4bit",
        "low_memory": True,
        "low_ram_streaming": args.low_ram,
    }
    pipeline_type = getattr(args, "pipeline_type", "distilled")

    if args.mode in ("retake", "extend"):
        return RetakePipeline(model_dir, **common)
    elif args.mode == "a2v":
        # A2V is its own two-stage Euler+CFG pipeline; it ignores pipeline_type.
        return A2VidPipelineTwoStage(model_dir, **common)
    elif pipeline_type == "two-stage":
        return TI2VidTwoStagesPipeline(model_dir, **common)
    elif pipeline_type == "two-stage-hq":
        return TI2VidTwoStagesHQPipeline(model_dir, **common)
    elif pipeline_type == "one-stage":
        return TI2VidOneStagePipeline(model_dir, **common)
    else:  # distilled (fastest, 8+3 steps, no CFG)
        return DistilledPipeline(model_dir, **common)


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

def _set_loras(pipeline, args: argparse.Namespace) -> None:
    """Set pending LoRAs on the pipeline before loading."""
    if args.lora:
        pipeline._pending_loras = _parse_lora_args(args.lora)


def _run_t2v(pipeline, args: argparse.Namespace) -> None:
    """Text-to-video or image-to-video generation."""
    global _current_stage

    _progress("STATUS:Loading model")
    _report_memory("before_load")

    _set_loras(pipeline, args)
    pipeline.load()

    _report_memory("after_model_load")
    _current_stage = 1
    _progress("STATUS:Generating video")

    pipeline_type = getattr(args, "pipeline_type", "distilled")

    gen_kwargs: dict = {
        "prompt": args.prompt,
        "output_path": args.output_path,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "frame_rate": float(args.fps),
        "seed": args.seed,
    }

    if pipeline_type == "one-stage":
        gen_kwargs["num_steps"] = args.num_steps
        gen_kwargs["cfg_scale"] = args.cfg_scale
        gen_kwargs["stg_scale"] = args.stg_scale
    elif pipeline_type in ("two-stage", "two-stage-hq"):
        gen_kwargs["stage1_steps"] = args.num_steps
        gen_kwargs["cfg_scale"] = args.cfg_scale
        gen_kwargs["stg_scale"] = args.stg_scale
    else:  # distilled — fixed sigma schedules, CFG/STG don't apply
        gen_kwargs["stage1_steps"] = args.num_steps

    if args.mode == "i2v" and args.image:
        # Pass the reference image with its conditioning strength. The library's
        # ``image=`` shorthand hardcodes strength=1.0, so to honor a user-set
        # strength we build the ImageConditioningInput explicitly. strength=1.0
        # reproduces the shorthand exactly.
        from ltx_pipelines_mlx.utils.args import ImageConditioningInput

        gen_kwargs["images"] = [
            ImageConditioningInput(
                path=args.image, frame_idx=0, strength=args.image_strength,
            )
        ]

    pipeline.generate_and_save(**gen_kwargs)

    _report_memory("after_generation")
    _progress("STATUS:Done")


def _run_a2v(pipeline, args: argparse.Namespace) -> None:
    """Audio-to-video generation (two-stage Euler + CFG, beta)."""
    global _current_stage

    _progress("STATUS:Loading model")
    _report_memory("before_load")

    _set_loras(pipeline, args)
    pipeline.load()

    _report_memory("after_model_load")
    _current_stage = 1
    _progress("STATUS:Generating video")

    gen_kwargs: dict = {
        "prompt": args.prompt,
        "output_path": args.output_path,
        "audio_path": args.audio,
        "audio_start_time": args.audio_start,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "frame_rate": float(args.fps),
        "seed": args.seed,
        "stage1_steps": args.num_steps,
        "cfg_scale": args.cfg_scale,
        "stg_scale": args.stg_scale,
    }

    # Optional reference image — A2V also supports I2V conditioning.
    if args.image:
        from ltx_pipelines_mlx.utils.args import ImageConditioningInput

        gen_kwargs["images"] = [
            ImageConditioningInput(
                path=args.image, frame_idx=0, strength=args.image_strength,
            )
        ]

    pipeline.generate_and_save(**gen_kwargs)

    _report_memory("after_generation")
    _progress("STATUS:Done")


def _decode_and_save(pipeline, video_latent, audio_latent, args: argparse.Namespace) -> None:
    """Decode latents and save, mirroring the library CLI's retake/extend flow.

    Frees the DiT + text encoder first so the decoders fit in memory, then
    loads decoders on demand.
    """
    if pipeline.low_memory:
        pipeline.dit = None
        pipeline.text_encoder = None
        pipeline.feature_extractor = None
        pipeline._loaded = False
        aggressive_cleanup()

    pipeline._load_decoders()
    pipeline._decode_and_save_video(
        video_latent, audio_latent, args.output_path, frame_rate=float(args.fps),
    )


def _run_retake(pipeline, args: argparse.Namespace) -> None:
    """Retake: regenerate a frame range in an existing video."""
    _progress("STATUS:Loading model")
    _report_memory("before_load")

    _set_loras(pipeline, args)
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
        cfg_scale=args.cfg_scale,
        stg_scale=args.stg_scale,
    )

    _progress("STATUS:Decoding video")
    _decode_and_save(pipeline, video_latent, audio_latent, args)

    _report_memory("after_generation")
    _progress("STATUS:Done")


def _run_extend(pipeline, args: argparse.Namespace) -> None:
    """Extend: add frames before or after an existing video."""
    _progress("STATUS:Loading model")
    _report_memory("before_load")

    _set_loras(pipeline, args)
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
        cfg_scale=args.cfg_scale,
        stg_scale=args.stg_scale,
    )

    _progress("STATUS:Decoding video")
    _decode_and_save(pipeline, video_latent, audio_latent, args)

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

    parser.add_argument("--mode", choices=["t2v", "i2v", "a2v", "retake", "extend", "enhance"],
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
    parser.add_argument("--pipeline-type",
                        choices=["distilled", "one-stage", "two-stage", "two-stage-hq"],
                        default="distilled", help="Pipeline variant")
    parser.add_argument("--cfg-scale", type=float, default=3.0,
                        help="CFG guidance scale (ignored by distilled)")
    parser.add_argument("--stg-scale", type=float, default=1.0,
                        help="STG guidance scale (ignored by distilled; 1.0 = upstream default)")
    parser.add_argument("--low-ram", action="store_true",
                        help="Stream DiT blocks from disk (~75%% less transformer RAM)")

    # I2V
    parser.add_argument("--image", default=None, help="Reference image path for I2V")
    parser.add_argument("--image-strength", type=float, default=1.0)

    # A2V
    parser.add_argument("--audio", default=None, help="Reference audio path for A2V")
    parser.add_argument("--audio-start", type=float, default=0.0,
                        help="Audio start time in seconds")

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

    _install_tqdm_hook()
    pipeline = _create_pipeline(args)

    if args.mode == "retake":
        _run_retake(pipeline, args)
    elif args.mode == "extend":
        _run_extend(pipeline, args)
    elif args.mode == "a2v":
        _run_a2v(pipeline, args)
    else:
        _run_t2v(pipeline, args)

    # Final cleanup
    del pipeline
    aggressive_cleanup()


if __name__ == "__main__":
    main()
