"""Async subprocess wrapper for LTX-2.3 MLX video generation.

Runs LTX-2.3 inference via ``python -m engine.generate_v23`` in a
subprocess, parsing stderr progress lines in real time and forwarding
them to an optional async callback.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections.abc import Awaitable, Callable
from pathlib import Path

from engine.memory_manager import aggressive_cleanup

log = logging.getLogger(__name__)

# Default model repo on HuggingFace (int8 quantized, MLX split format)
DEFAULT_MODEL_REPO = "dgrauet/ltx-2.3-mlx-distilled-q8"

# HuggingFace cache root
_HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

# Legacy local path (from earlier local_dir downloads — checked as fallback)
_LEGACY_LOCAL_PATH = _HF_CACHE / "ltx23-mlx"

# Regex patterns for stderr progress lines
_STAGE_RE = re.compile(r"^STAGE:(\d+):STEP:(\d+):(\d+)")
_STATUS_RE = re.compile(r"^STATUS:(.+)")
_MEMORY_RE = re.compile(r"^MEMORY:(\w+):active=([\d.]+):cache=([\d.]+):peak=([\d.]+)")
_PREVIEW_RE = re.compile(r"^PREVIEW:(.+)")

# Progress mapping — maps (stage, step/total) to overall 0.0-1.0 range
# When two-stage is active, Stage 1 gets 0.05-0.55, Stage 2 gets 0.65-0.80
# When single-stage, Stage 1 gets 0.05-0.85 (same as before)
_STAGE_RANGES: dict[int, tuple[float, float]] = {
    1: (0.05, 0.55),  # Stage 1 denoising (half res)
    2: (0.65, 0.80),  # Stage 2 refinement (target res)
}

# Fallback for single-stage (no stage 2 progress lines emitted)
_SINGLE_STAGE_RANGES: dict[int, tuple[float, float]] = {
    1: (0.05, 0.85),
}


def _is_quantized_model(model_path: Path) -> bool:
    """Check if a model directory contains quantized weights."""
    qconfig = model_path / "quantize_config.json"
    return qconfig.exists()


def get_model_repo() -> tuple[str, bool]:
    """Return the model path and whether it's quantized.

    Resolves the default HuggingFace repo from local cache, with fallback
    to a legacy local path for backwards compatibility.

    Returns:
        Tuple of (model_path, is_quantized).
    """
    # Try HF standard cache first
    model_path = _resolve_hf_model(DEFAULT_MODEL_REPO)
    if model_path:
        quantized = _is_quantized_model(Path(model_path))
        log.info("Using HF model: %s (%s)", DEFAULT_MODEL_REPO, model_path)
        return model_path, quantized

    # Fallback: legacy local path (from earlier local_dir downloads)
    if _LEGACY_LOCAL_PATH.exists() and (_LEGACY_LOCAL_PATH / "transformer.safetensors").exists():
        quantized = _is_quantized_model(_LEGACY_LOCAL_PATH)
        log.info("Using legacy local model: %s", _LEGACY_LOCAL_PATH)
        return str(_LEGACY_LOCAL_PATH), quantized

    # Last resort: return repo ID (generation will fail if not downloadable)
    log.warning("Could not resolve model %s — returning repo ID", DEFAULT_MODEL_REPO)
    return DEFAULT_MODEL_REPO, True


def _resolve_hf_model(repo_id: str) -> str | None:
    """Resolve a HuggingFace repo to a local cached path.

    Uses huggingface_hub to check if the model is already cached,
    without triggering a download. Returns the snapshot path if cached.

    Args:
        repo_id: HuggingFace repository ID.

    Returns:
        Local path to the cached model, or None if not cached.
    """
    try:
        from huggingface_hub import try_to_load_from_cache

        # Check that main model weights are cached (implies config is too)
        weights_cached = try_to_load_from_cache(repo_id, "transformer.safetensors")
        if weights_cached and isinstance(weights_cached, str):
            return str(Path(weights_cached).parent)
    except Exception as e:
        log.debug("Could not check HF cache for %s: %s", repo_id, e)
    return None


def get_venv_python() -> str:
    """Auto-detect the venv Python path relative to this file.

    Walks up from ``engine/mlx_runner.py`` to find ``backend/.venv/bin/python``.

    Returns:
        Absolute path to the venv Python binary.

    Raises:
        FileNotFoundError: If the venv Python binary cannot be found.
    """
    # __file__ is backend/engine/mlx_runner.py → parent.parent = backend/
    backend_dir = Path(__file__).resolve().parent.parent
    venv_python = backend_dir / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    raise FileNotFoundError(
        f"Venv Python not found at {venv_python}. "
        f"Run 'uv sync' in {backend_dir} to create the virtual environment."
    )


def _get_text_encoder_4bit() -> str | None:
    """Return the path to a 4-bit text encoder if available.

    On 32GB machines, the bf16 Gemma 3 12B (~24GB) is too large.
    Uses the 4-bit version (~6GB) instead.
    """
    # Check for locally cached 4-bit text encoder
    model_dir = _HF_CACHE / "models--mlx-community--gemma-3-12b-it-4bit"
    if model_dir.exists():
        snapshots = model_dir / "snapshots"
        if snapshots.exists():
            for d in sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                if (d / "config.json").exists():
                    log.info("Using 4-bit text encoder: %s", d)
                    return str(d)
    # Fallback: use the HF repo ID (will download on first use)
    return "mlx-community/gemma-3-12b-it-4bit"



def _get_upscaler_weights() -> str | None:
    """Find or download the LTX-2.3 spatial upscaler weights.

    Searches local HuggingFace cache for the upscaler safetensors file.
    Falls back to downloading from HuggingFace if not found locally.

    Returns:
        Path to the upscaler weights file, or None if unavailable.
    """
    _UPSCALER_REPO = "Lightricks/LTX-2.3"
    _UPSCALER_FILE = "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"

    # Check HF cache for upscaler weights inside main LTX-2.3 repo
    repo_dir = _HF_CACHE / "models--Lightricks--LTX-2.3" / "snapshots"
    if repo_dir.exists():
        for snapshot in sorted(repo_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            candidate = snapshot / _UPSCALER_FILE
            if candidate.exists():
                log.info("Found upscaler weights: %s", candidate)
                return str(candidate)

    # Not cached locally — download just the upscaler file from the main repo
    try:
        from huggingface_hub import hf_hub_download

        log.info("Downloading upscaler weights from %s...", _UPSCALER_REPO)
        path = hf_hub_download(
            repo_id=_UPSCALER_REPO,
            filename=_UPSCALER_FILE,
        )
        log.info("Downloaded upscaler weights: %s", path)
        return path
    except Exception as e:
        log.warning("Could not find or download upscaler weights: %s", e)
        return None


def _compute_progress(stage: int, step: int, total: int, is_two_stage: bool = False) -> float:
    """Compute overall progress percentage from stage/step info.

    Args:
        stage: Current stage number (1 or 2).
        step: Current step within the stage.
        total: Total steps in the stage.
        is_two_stage: If True, use two-stage progress ranges.

    Returns:
        Overall progress as a float between 0.0 and 1.0.
    """
    ranges = _STAGE_RANGES if is_two_stage else _SINGLE_STAGE_RANGES
    if stage in ranges:
        lo, hi = ranges[stage]
        frac = step / max(total, 1)
        return lo + frac * (hi - lo)
    # Unknown stage — return rough estimate
    return 0.85


async def _run_text_encoding_subprocess(
    python_bin: str,
    model_repo: str,
    prompt: str,
    embeddings_path: str,
    backend_dir: str,
    text_encoder_repo: str | None = None,
) -> None:
    """Run text encoding in an isolated subprocess.

    On 32GB machines, the text encoder (4-bit Gemma 3 12B ~7.5GB + connectors
    ~2.7GB) and the video transformer (~10GB) cannot coexist. By encoding in a
    separate subprocess that exits before generation starts, we guarantee the
    text encoder memory is fully freed.
    """
    cmd = [
        python_bin,
        "-m", "engine.encode_text_subprocess",
        "--prompt", prompt,
        "--model-repo", model_repo,
        "--output", embeddings_path,
    ]
    if text_encoder_repo:
        cmd.extend(["--text-encoder-repo", text_encoder_repo])

    log.info("Running text encoding subprocess...")
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=backend_dir,
    )

    _, stderr_data = await proc.communicate()
    stderr_text = stderr_data.decode("utf-8", errors="replace")

    if proc.returncode != 0:
        log.error("Text encoding failed (exit code %d): %s", proc.returncode, stderr_text)
        if proc.returncode == -6:
            raise RuntimeError(
                "Not enough GPU memory for text encoding. "
                "Close other apps and try again."
            )
        raise RuntimeError(
            f"Text encoding failed with exit code {proc.returncode}:\n{stderr_text}"
        )

    for line in stderr_text.strip().split("\n"):
        if line.strip():
            log.debug("encode_text: %s", line)

    log.info("Text encoding complete, embeddings saved to %s", embeddings_path)


async def run_mlx_generation(
    prompt: str,
    height: int,
    width: int,
    num_frames: int,
    seed: int,
    fps: int,
    output_path: str,
    image: str | None = None,
    image_strength: float = 1.0,
    num_steps: int = 8,
    upscale: bool = False,
    ffmpeg_upscale: bool = False,
    preview_interval: int = 0,
    skip_bwe: bool = True,
    lora_args: list[str] | None = None,
    progress_callback: Callable[..., Awaitable[None]] | None = None,
    venv_python: str | None = None,
) -> dict:
    """Run LTX-2.3 video generation as an async subprocess.

    Text encoding runs in a separate subprocess first (for 32GB memory
    management). Embeddings are passed via LTX_PRECOMPUTED_EMBEDDINGS env var.

    Args:
        prompt: Text prompt describing the video to generate.
        height: Output video height in pixels.
        width: Output video width in pixels.
        num_frames: Number of frames to generate.
        seed: Random seed for reproducibility.
        fps: Output frames per second.
        output_path: Path where the output MP4 will be written.
        image: Optional path to a reference image for I2V generation.
        image_strength: Strength of image conditioning (0.0-1.0).
        num_steps: Number of denoising steps (default 8 for distilled model).
        upscale: If True, use two-stage neural upscale pipeline.
        ffmpeg_upscale: If True, apply ffmpeg lanczos 2x post-processing.
        preview_interval: Emit preview frames every N diffusion steps (0=off).
        skip_bwe: If True, disable bandwidth extension (16kHz audio).
        lora_args: Optional LoRA arguments (--lora path:strength).
        progress_callback: Optional async callback invoked with
            ``(step, total_steps, stage, pct, status=None)`` where ``pct``
            is 0.0-1.0 and ``status`` is an optional human-readable stage label.
        venv_python: Path to venv Python binary. Auto-detected if None.

    Returns:
        Dictionary with ``output_path`` and ``subprocess_memory``.

    Raises:
        RuntimeError: If the subprocess exits with a non-zero code.
        FileNotFoundError: If the venv Python binary cannot be found.
    """
    python_bin = venv_python or get_venv_python()
    model_repo, is_quantized = get_model_repo()
    module = "engine.generate_v23"
    log.info("Using LTX-2.3 vendored pipeline")

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Run from backend/ directory so engine.* imports work for quantized wrapper
    backend_dir = str(Path(__file__).resolve().parent.parent)

    # Get 4-bit text encoder path
    text_encoder_4bit = _get_text_encoder_4bit()

    # For quantized models, run text encoding in a separate subprocess
    # to avoid OOM when both text encoder and transformer compete for GPU memory
    embeddings_path: str | None = None
    if is_quantized:
        import tempfile
        embeddings_dir = Path(output_path).parent
        embeddings_path = str(embeddings_dir / f"_embeddings_{Path(output_path).stem}.npz")

        if progress_callback is not None:
            await progress_callback(0, 0, 0, 0.01, status="Encoding text")

        await _run_text_encoding_subprocess(
            python_bin=python_bin,
            model_repo=model_repo,
            prompt=prompt,
            embeddings_path=embeddings_path,
            backend_dir=backend_dir,
            text_encoder_repo=text_encoder_4bit,
        )

        if progress_callback is not None:
            await progress_callback(0, 0, 0, 0.05, status="Text encoding complete")

    cmd = [
        python_bin,
        "-m", module,
        "--prompt", prompt,
        "--height", str(height),
        "--width", str(width),
        "--num-frames", str(num_frames),
        "--seed", str(seed),
        "--fps", str(fps),
        "--output-path", output_path,
        "--model-dir", model_repo,
        "--num-steps", str(num_steps),
        "--generate-audio",
    ]

    if image is not None:
        cmd.extend(["--image", image, "--image-strength", str(image_strength)])

    # Append LoRA arguments (--lora path:strength, can be repeated)
    if lora_args:
        cmd.extend(lora_args)

    is_two_stage = False
    if upscale:
        upscaler_path = _get_upscaler_weights()
        if upscaler_path:
            cmd.extend(["--upscale", upscaler_path])
            is_two_stage = True
            log.info("Two-stage upscale enabled: %s", upscaler_path)
        else:
            cmd.append("--ffmpeg-upscale")
            log.info("Neural upscaler weights not found, falling back to ffmpeg lanczos 2x")

    if ffmpeg_upscale:
        cmd.append("--ffmpeg-upscale")
        log.info("ffmpeg lanczos 2x post-processing enabled")

    if preview_interval > 0:
        cmd.extend(["--preview-interval", str(preview_interval)])
        log.info("Preview frames enabled every %d steps", preview_interval)

    if skip_bwe:
        cmd.append("--no-bwe")
        log.info("BWE disabled — audio output at 16kHz (less metallic)")


    log.info("Starting MLX generation: %s", " ".join(cmd[:6]) + " ...")
    log.debug("Full command: %s", cmd)

    # Set up environment with precomputed embeddings path
    env = None
    if embeddings_path:
        env = os.environ.copy()
        env["LTX_PRECOMPUTED_EMBEDDINGS"] = embeddings_path

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=backend_dir,
        env=env,
    )

    stderr_lines: list[str] = []
    # Track memory stats reported by the subprocess
    subprocess_memory: dict[str, dict[str, float]] = {}
    # Track last progress values for PREVIEW lines
    last_step, last_total, last_stage, last_pct = 0, 0, 0, 0.0
    last_status: str | None = "Generating video"

    # Read stderr line by line for progress parsing
    assert proc.stderr is not None  # noqa: S101
    while True:
        raw_line = await proc.stderr.readline()
        if not raw_line:
            break
        line = raw_line.decode("utf-8", errors="replace").rstrip()
        stderr_lines.append(line)

        # Parse PREVIEW lines (file path to JPEG preview frame)
        preview_match = _PREVIEW_RE.match(line)
        if preview_match:
            preview_path = preview_match.group(1).strip()
            try:
                import base64

                with open(preview_path, "rb") as f:
                    b64_frame = base64.b64encode(f.read()).decode("ascii")
                os.unlink(preview_path)
                log.debug("Preview frame: %d bytes base64", len(b64_frame))
                if progress_callback is not None:
                    await progress_callback(
                        last_step, last_total, last_stage, last_pct,
                        status=last_status, preview_frame=b64_frame,
                    )
            except Exception as e:
                log.warning("Failed to read preview frame %s: %s", preview_path, e)
            continue

        # Parse STAGE:X:STEP:Y:TOTAL lines
        stage_match = _STAGE_RE.match(line)
        if stage_match:
            stage = int(stage_match.group(1))
            step = int(stage_match.group(2))
            total = int(stage_match.group(3))
            pct = _compute_progress(stage, step, total, is_two_stage=is_two_stage)
            last_step, last_total, last_stage, last_pct = step, total, stage, pct
            last_status = "Generating video"
            log.debug("Progress: stage=%d step=%d/%d pct=%.2f", stage, step, total, pct)
            if progress_callback is not None:
                await progress_callback(step, total, stage, pct, status="Generating video")
            continue

        # Parse MEMORY lines (e.g. MEMORY:after_model_load:active=12.5:cache=2.1:peak=14.3)
        memory_match = _MEMORY_RE.match(line)
        if memory_match:
            label = memory_match.group(1)
            subprocess_memory[label] = {
                "active_memory_gb": float(memory_match.group(2)),
                "cache_memory_gb": float(memory_match.group(3)),
                "peak_memory_gb": float(memory_match.group(4)),
            }
            log.info(
                "Subprocess memory [%s]: active=%.3f GB, cache=%.3f GB, peak=%.3f GB",
                label,
                subprocess_memory[label]["active_memory_gb"],
                subprocess_memory[label]["cache_memory_gb"],
                subprocess_memory[label]["peak_memory_gb"],
            )
            continue

        # Parse STATUS lines
        status_match = _STATUS_RE.match(line)
        if status_match:
            status_msg = status_match.group(1).strip()
            last_status = status_msg
            log.info("MLX status: %s", status_msg)

            # Map status messages to progress percentages
            pct = 0.85
            if "stage 1" in status_msg.lower():
                pct = 0.06
            elif "upscaling latent" in status_msg.lower():
                pct = 0.57
            elif "reloading model" in status_msg.lower():
                pct = 0.60
            elif "stage 2" in status_msg.lower():
                pct = 0.63
            elif "upscaling 2x (ffmpeg)" in status_msg.lower():
                pct = 0.95
            elif "upscaling" in status_msg.lower():
                pct = 0.83
            elif "decoding video" in status_msg.lower():
                pct = 0.88
            elif "decoding audio" in status_msg.lower():
                pct = 0.93
            elif "saving" in status_msg.lower():
                pct = 0.97
            elif "loading" in status_msg.lower():
                pct = 0.02

            if progress_callback is not None:
                await progress_callback(0, 0, 0, pct, status=status_msg)
            continue

        # Log other stderr lines at debug level
        if line:
            log.debug("MLX stderr: %s", line)

    # Wait for process to complete
    await proc.wait()

    # Cleanup temp embeddings file
    if embeddings_path:
        try:
            Path(embeddings_path).unlink(missing_ok=True)
        except OSError:
            pass

    # Cleanup Metal memory after generation
    aggressive_cleanup()

    if proc.returncode != 0:
        stderr_text = "\n".join(stderr_lines[-100:])  # Last 100 lines for context
        log.error("MLX generation failed (exit code %d): %s", proc.returncode, stderr_text)

        # Detect Metal GPU out-of-memory crash (exit code -6 = SIGABRT from Metal)
        if proc.returncode == -6 and "Command buffer execution failed" in stderr_text:
            raise RuntimeError(
                "Not enough GPU memory. Close other apps (including Xcode) and try again, "
                "or use a lower resolution / fewer frames."
            )

        raise RuntimeError(
            f"MLX generation failed with exit code {proc.returncode}:\n{stderr_text}"
        )

    # Signal completion
    if progress_callback is not None:
        await progress_callback(0, 0, 0, 1.0, status="Complete")

    log.info("MLX generation complete: %s", output_path)
    return {
        "output_path": output_path,
        "subprocess_memory": subprocess_memory,
    }
