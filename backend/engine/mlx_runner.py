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

# Quantized model paths (checked in priority order)
_QUANTIZED_PATHS = [
    Path.home() / ".cache/huggingface/hub/ltx2-mlx-av-int4",
    Path.home() / ".cache/huggingface/hub/ltx2-mlx-av-int8",
]

# Regex patterns for stderr progress lines
_STAGE_RE = re.compile(r"^STAGE:(\d+):STEP:(\d+):(\d+)")
_STATUS_RE = re.compile(r"^STATUS:(.+)")

# Progress mapping — maps (stage, step/total) to overall 0.0-1.0 range
_STAGE_RANGES: dict[int, tuple[float, float]] = {
    1: (0.05, 0.50),  # Stage 1 denoising
    2: (0.50, 0.85),  # Stage 2 denoising
}


def _is_quantized_model(model_path: Path) -> bool:
    """Check if a model directory contains quantized weights."""
    qconfig = model_path / "quantize_config.json"
    return qconfig.exists()


def get_model_repo() -> tuple[str, bool]:
    """Return the best available model and whether it's quantized.

    Checks for local quantized models (int4 first, then int8). If found,
    returns the local directory path. Otherwise returns the default HF repo ID.

    Returns:
        Tuple of (model_repo_or_path, is_quantized).
    """
    for qpath in _QUANTIZED_PATHS:
        config_file = qpath / "config.json"
        # Check for either monolithic or split model files
        has_model = (qpath / "model.safetensors").exists() or (qpath / "transformer.safetensors").exists()
        if has_model and config_file.exists():
            quantized = _is_quantized_model(qpath)
            log.info("Using %s model: %s", "quantized" if quantized else "local", qpath)
            return str(qpath), quantized
    log.info("Using default model: %s", DEFAULT_MODEL_REPO)
    return DEFAULT_MODEL_REPO, False


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


def _compute_progress(stage: int, step: int, total: int) -> float:
    """Compute overall progress percentage from stage/step info.

    Args:
        stage: Current stage number (1 or 2).
        step: Current step within the stage.
        total: Total steps in the stage.

    Returns:
        Overall progress as a float between 0.0 and 1.0.
    """
    if stage in _STAGE_RANGES:
        lo, hi = _STAGE_RANGES[stage]
        frac = step / max(total, 1)
        return lo + frac * (hi - lo)
    # Unknown stage — return rough estimate
    return 0.85


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
    progress_callback: Callable[[int, int, int, float], Awaitable[None]] | None = None,
    venv_python: str | None = None,
) -> str:
    """Run MLX video generation as an async subprocess.

    Launches ``python -m mlx_video.generate_av`` with the given parameters,
    parses progress from stderr in real time, and calls ``progress_callback``
    for each update.

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
        progress_callback: Optional async callback invoked with
            ``(step, total_steps, stage, pct)`` where ``pct`` is 0.0-1.0.
        venv_python: Path to venv Python binary. Auto-detected if None.

    Returns:
        The ``output_path`` string on success.

    Raises:
        RuntimeError: If the subprocess exits with a non-zero code.
        FileNotFoundError: If the venv Python binary cannot be found.
    """
    python_bin = venv_python or get_venv_python()
    model_repo, is_quantized = get_model_repo()

    # Use quantized wrapper module for quantized models, otherwise standard mlx_video
    if is_quantized:
        module = "engine.generate_av_quantized"
        log.info("Using quantized inference wrapper")
    else:
        module = "mlx_video.generate_av"

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

    # Use 4-bit text encoder on memory-constrained machines
    text_encoder_4bit = _get_text_encoder_4bit()
    if text_encoder_4bit:
        cmd.extend(["--text-encoder-repo", text_encoder_4bit])

    if image is not None:
        cmd.extend(["--image", image, "--image-strength", str(image_strength)])

    log.info("Starting MLX generation: %s", " ".join(cmd[:6]) + " ...")
    log.debug("Full command: %s", cmd)

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Run from backend/ directory so engine.* imports work for quantized wrapper
    backend_dir = str(Path(__file__).resolve().parent.parent)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=backend_dir,
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
            pct = _compute_progress(stage, step, total)
            log.debug("Progress: stage=%d step=%d/%d pct=%.2f", stage, step, total, pct)
            if progress_callback is not None:
                await progress_callback(step, total, stage, pct)
            continue

        # Parse STATUS lines
        status_match = _STATUS_RE.match(line)
        if status_match:
            status_msg = status_match.group(1).strip()
            log.info("MLX status: %s", status_msg)

            # Map status messages to progress percentages
            pct = 0.85
            if "decoding video" in status_msg.lower():
                pct = 0.88
            elif "decoding audio" in status_msg.lower():
                pct = 0.93
            elif "saving" in status_msg.lower():
                pct = 0.97
            elif "loading" in status_msg.lower():
                pct = 0.02

            if progress_callback is not None:
                await progress_callback(0, 0, 0, pct)
            continue

        # Log other stderr lines at debug level
        if line:
            log.debug("MLX stderr: %s", line)

    # Wait for process to complete
    await proc.wait()

    # Cleanup Metal memory after generation
    aggressive_cleanup()

    if proc.returncode != 0:
        stderr_text = "\n".join(stderr_lines[-50:])  # Last 50 lines for context
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
        await progress_callback(0, 0, 0, 1.0)

    log.info("MLX generation complete: %s", output_path)
    return output_path
