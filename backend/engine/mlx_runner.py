"""Async subprocess wrapper for MLX video generation.

Runs real MLX inference via ``python -m mlx_video.generate_av`` in a
subprocess, parsing stderr progress lines in real time and forwarding
them to an optional async callback.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from pathlib import Path

from engine.memory_manager import aggressive_cleanup

log = logging.getLogger(__name__)

# Default model repo — overridden if a local quantized model is detected
DEFAULT_MODEL_REPO = "notapalindrome/ltx2-mlx-av"

# Quantized model paths (checked in priority order — v2.3 preferred, then v2.0 int8/int4)
_QUANTIZED_PATHS = [
    Path.home() / ".cache/huggingface/hub/ltx23-mlx",
    Path.home() / ".cache/huggingface/hub/ltx2-mlx-av-int8",
    Path.home() / ".cache/huggingface/hub/ltx2-mlx-av-int4",
]

# Regex patterns for stderr progress lines
_STAGE_RE = re.compile(r"^STAGE:(\d+):STEP:(\d+):(\d+)")
_STATUS_RE = re.compile(r"^STATUS:(.+)")

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


def _get_model_version(model_path: Path) -> str:
    """Detect model version from config.json.

    Returns:
        "2.3" for LTX-2.3, "2.0" for LTX-2.0.
    """
    config_file = model_path / "config.json"
    if config_file.exists():
        import json
        with open(config_file) as f:
            config = json.load(f)
        version = config.get("model_version", "")
        if version.startswith("2.3") or config.get("is_v2", False):
            return "2.3"
    return "2.0"


def get_model_repo() -> tuple[str, bool, str]:
    """Return the best available model, whether it's quantized, and its version.

    Checks for local models in priority order: LTX-2.3, then LTX-2.0 int8/int4.
    If no local model found, returns the default HF repo ID.

    Returns:
        Tuple of (model_repo_or_path, is_quantized, version).
        version is "2.3" or "2.0".
    """
    for qpath in _QUANTIZED_PATHS:
        config_file = qpath / "config.json"
        # Check for either monolithic or split model files
        has_model = (qpath / "model.safetensors").exists() or (qpath / "transformer.safetensors").exists()
        if has_model and config_file.exists():
            quantized = _is_quantized_model(qpath)
            version = _get_model_version(qpath)
            log.info(
                "Using %s model v%s: %s",
                "quantized" if quantized else "local",
                version,
                qpath,
            )
            return str(qpath), quantized, version
    log.info("Using default model: %s", DEFAULT_MODEL_REPO)
    return DEFAULT_MODEL_REPO, False, "2.0"


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
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = cache_root / "models--mlx-community--gemma-3-12b-it-4bit"
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
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"

    # Check LTX-2.3 repo first
    for repo_name in ["models--Lightricks--LTX-2.3", "models--Lightricks--LTX-2"]:
        repo_dir = cache_root / repo_name / "snapshots"
        if not repo_dir.exists():
            continue
        for snapshot in sorted(repo_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            for fname in [
                "ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
                "ltx-2-spatial-upscaler-x2-1.0.safetensors",
            ]:
                candidate = snapshot / fname
                if candidate.exists():
                    log.info("Found upscaler weights: %s", candidate)
                    return str(candidate)

    # Not cached locally — try downloading
    try:
        from huggingface_hub import hf_hub_download

        log.info("Downloading upscaler weights from Lightricks/LTX-2.3...")
        path = hf_hub_download(
            repo_id="Lightricks/LTX-2.3",
            filename="ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
        )
        log.info("Downloaded upscaler weights: %s", path)
        return path
    except Exception:
        # Try LTX-2 as fallback
        try:
            path = hf_hub_download(
                repo_id="Lightricks/LTX-2",
                filename="ltx-2-spatial-upscaler-x2-1.0.safetensors",
            )
            log.info("Downloaded upscaler weights (LTX-2): %s", path)
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
    tiling: str = "auto",
    upscale: bool = False,
    progress_callback: Callable[..., Awaitable[None]] | None = None,
    venv_python: str | None = None,
) -> str:
    """Run MLX video generation as an async subprocess.

    For quantized models on 32GB machines, text encoding runs in a separate
    subprocess first. The embeddings are saved to a temp file and passed to
    the generation subprocess via the LTX_PRECOMPUTED_EMBEDDINGS env var.

    When upscale is True and running v2.3, the two-stage neural upscale
    pipeline is used: generate at half resolution, upscale latent 2x with
    the learned LatentUpsampler, then refine at target resolution.

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
        tiling: Tiling mode (``"auto"``, ``"on"``, ``"off"``).
        upscale: If True, use two-stage neural upscale pipeline (v2.3) or
            ffmpeg lanczos upscale (v2.0).
        progress_callback: Optional async callback invoked with
            ``(step, total_steps, stage, pct, status=None)`` where ``pct``
            is 0.0-1.0 and ``status`` is an optional human-readable stage label.
        venv_python: Path to venv Python binary. Auto-detected if None.

    Returns:
        The ``output_path`` string on success.

    Raises:
        RuntimeError: If the subprocess exits with a non-zero code.
        FileNotFoundError: If the venv Python binary cannot be found.
    """
    python_bin = venv_python or get_venv_python()
    model_repo, is_quantized, model_version = get_model_repo()

    # Route to appropriate generation module based on model version
    if model_version == "2.3":
        module = "engine.generate_v23"
        log.info("Using LTX-2.3 vendored pipeline")
    elif is_quantized:
        module = "engine.generate_av_quantized"
        log.info("Using quantized inference wrapper (LTX-2.0)")
    else:
        module = "mlx_video.generate_av"

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

    if model_version == "2.3":
        # LTX-2.3 uses --model-dir instead of --model-repo
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
            "--generate-audio",
        ]
    else:
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
            "--model-repo", model_repo,
            "--tiling", tiling,
        ]
        # Use 4-bit text encoder (only needed if not using precomputed embeddings)
        if text_encoder_4bit and not embeddings_path:
            cmd.extend(["--text-encoder-repo", text_encoder_4bit])

    if image is not None:
        cmd.extend(["--image", image, "--image-strength", str(image_strength)])

    is_two_stage = False
    if upscale and model_version == "2.3":
        # Two-stage neural upscale: find upsampler weights and pass path
        upscaler_path = _get_upscaler_weights()
        if upscaler_path:
            cmd.extend(["--upscale", upscaler_path])
            is_two_stage = True
            log.info("Two-stage upscale enabled: %s", upscaler_path)
        else:
            log.warning("Upscaler weights not found, falling back to single-stage")
    elif upscale:
        # LTX-2.0 fallback: ffmpeg lanczos (--upscale without path not supported)
        log.info("Upscale requested for v2.0 — not supported, skipping")

    log.info("Starting MLX generation: %s", " ".join(cmd[:6]) + " ...")
    log.debug("Full command: %s", cmd)

    # Set up environment with precomputed embeddings path
    env = None
    if embeddings_path:
        import os
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

    # Read stderr line by line for progress parsing
    assert proc.stderr is not None  # noqa: S101
    while True:
        raw_line = await proc.stderr.readline()
        if not raw_line:
            break
        line = raw_line.decode("utf-8", errors="replace").rstrip()
        stderr_lines.append(line)

        # Parse STAGE:X:STEP:Y:TOTAL lines
        stage_match = _STAGE_RE.match(line)
        if stage_match:
            stage = int(stage_match.group(1))
            step = int(stage_match.group(2))
            total = int(stage_match.group(3))
            pct = _compute_progress(stage, step, total, is_two_stage=is_two_stage)
            log.debug("Progress: stage=%d step=%d/%d pct=%.2f", stage, step, total, pct)
            if progress_callback is not None:
                await progress_callback(step, total, stage, pct, status="Generating video")
            continue

        # Parse STATUS lines
        status_match = _STATUS_RE.match(line)
        if status_match:
            status_msg = status_match.group(1).strip()
            log.info("MLX status: %s", status_msg)

            # Map status messages to progress percentages
            pct = 0.85
            if "stage 1" in status_msg.lower():
                pct = 0.06
            elif "upscaling latent" in status_msg.lower():
                pct = 0.57
            elif "stage 2" in status_msg.lower():
                pct = 0.63
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
    return output_path
