"""Async subprocess orchestrator for MLX generation.

Launches ``python -m engine.generate_v23`` as a subprocess and parses its
stderr output for real-time progress reporting back to the FastAPI server.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
from collections.abc import Awaitable, Callable
from pathlib import Path

from huggingface_hub import try_to_load_from_cache

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_REPO = "dgrauet/ltx-2.3-mlx-q8"

# Fallback repo IDs for models cached under old names (before rename)
_REPO_ALIASES = {
    "dgrauet/ltx-2.3-mlx-q8": "dgrauet/ltx-2.3-mlx-distilled-q8",
    "dgrauet/ltx-2.3-mlx-q4": "dgrauet/ltx-2.3-mlx-distilled-q4",
    "dgrauet/ltx-2.3-mlx": "dgrauet/ltx-2.3-mlx-distilled",
}

_HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

# Stderr line patterns (must match generate_v23.py output)
_STAGE_RE = re.compile(r"^STAGE:(\d+):STEP:(\d+):(\d+)")
_STATUS_RE = re.compile(r"^STATUS:(.+)")
_MEMORY_RE = re.compile(r"^MEMORY:(\w+):active=([\d.]+):cache=([\d.]+):peak=([\d.]+)")
_PREVIEW_RE = re.compile(r"^PREVIEW:(.+)")

# Progress ranges for mapping STAGE lines to 0.0-1.0
_STAGE_RANGES = {1: (0.05, 0.55), 2: (0.65, 0.80)}
_SINGLE_STAGE_RANGES = {1: (0.05, 0.85)}

# Status string -> approximate progress
_STATUS_PROGRESS = {
    "loading": 0.02,
    "stage 1": 0.06,
    "generating": 0.10,
    "upscaling latent": 0.57,
    "reloading model": 0.60,
    "stage 2": 0.63,
    "upscaling": 0.83,
    "decoding video": 0.88,
    "decoding audio": 0.93,
    "retaking": 0.10,
    "extending": 0.10,
    "saving": 0.97,
    "done": 1.0,
}


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------

def get_model_repo(repo_id: str | None = None) -> tuple[str, bool]:
    """Resolve a HuggingFace repo ID to a local cache path.

    Returns:
        (model_path, is_quantized) -- path is the repo ID if not cached locally.
    """
    target_repo = repo_id or DEFAULT_MODEL_REPO

    model_path = _resolve_hf_model(target_repo)
    if model_path:
        quantized = _is_quantized_model(Path(model_path))
        log.info("Using HF model: %s (%s)", target_repo, model_path)
        return model_path, quantized

    log.warning("Could not resolve model %s -- returning repo ID", target_repo)
    return target_repo, True


def _resolve_hf_model(repo_id: str) -> str | None:
    """Check HF cache for a downloaded model. Returns directory path or None.

    Tries the given repo_id first, then falls back to old aliases
    (for models cached before the repo rename).
    """
    candidates = [repo_id]
    if repo_id in _REPO_ALIASES:
        candidates.append(_REPO_ALIASES[repo_id])

    for candidate in candidates:
        for check_file in ("transformer-distilled.safetensors", "transformer-dev.safetensors",
                           "transformer.safetensors"):
            result = try_to_load_from_cache(candidate, check_file)
            if result and isinstance(result, str):
                return str(Path(result).parent)
    return None


def _is_quantized_model(model_dir: Path) -> bool:
    """Check if model has quantization config."""
    return (model_dir / "quantize_config.json").exists()


def get_venv_python() -> str:
    """Auto-detect the venv Python binary."""
    backend_dir = Path(__file__).resolve().parent.parent
    venv_python = backend_dir / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    raise FileNotFoundError(f"No venv Python at {venv_python}")


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def _compute_progress(stage: int, step: int, total: int, *, two_stage: bool = False) -> float:
    """Map (stage, step, total) to a 0.0-1.0 progress value."""
    ranges = _STAGE_RANGES if two_stage else _SINGLE_STAGE_RANGES
    lo, hi = ranges.get(stage, (0.0, 1.0))
    if total <= 0:
        return lo
    frac = step / total
    return lo + frac * (hi - lo)


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

async def run_mlx_generation(
    prompt: str,
    height: int,
    width: int,
    num_frames: int,
    seed: int,
    fps: int,
    output_path: str,
    mode: str = "t2v",
    image: str | None = None,
    image_strength: float = 1.0,
    num_steps: int = 8,
    pipeline_type: str = "one-stage",
    cfg_scale: float = 3.0,
    stg_scale: float = 0.0,
    enhance_prompt: bool = False,
    lora_args: list[str] | None = None,
    retake_source: str | None = None,
    retake_start_frame: int = 0,
    retake_end_frame: int = -1,
    extend_source: str | None = None,
    extend_frames: int = 49,
    extend_direction: str = "after",
    progress_callback: Callable[..., Awaitable[None]] | None = None,
    venv_python: str | None = None,
    model_repo_id: str | None = None,
) -> dict:
    """Launch a generation subprocess and stream progress back.

    Returns:
        Dict with ``output_path`` and ``subprocess_memory`` snapshots.
    """
    python_bin = venv_python or get_venv_python()
    model_repo, _ = get_model_repo(model_repo_id)
    backend_dir = str(Path(__file__).resolve().parent.parent)

    # Build command
    cmd = [
        python_bin, "-m", "engine.generate_v23",
        "--mode", mode,
        "--prompt", prompt,
        "--model-dir", model_repo,
        "--height", str(height),
        "--width", str(width),
        "--num-frames", str(num_frames),
        "--seed", str(seed),
        "--fps", str(fps),
        "--output-path", output_path,
        "--num-steps", str(num_steps),
        "--pipeline-type", pipeline_type,
        "--cfg-scale", str(cfg_scale),
        "--stg-scale", str(stg_scale),
    ]

    # I2V args
    if image:
        cmd.extend(["--image", image, "--image-strength", str(image_strength)])

    # Retake args
    if retake_source:
        cmd.extend([
            "--retake-source", retake_source,
            "--retake-start-frame", str(retake_start_frame),
            "--retake-end-frame", str(retake_end_frame),
        ])

    # Extend args
    if extend_source:
        cmd.extend([
            "--extend-source", extend_source,
            "--extend-frames", str(extend_frames),
            "--extend-direction", extend_direction,
        ])

    # LoRA args
    if lora_args:
        for la in lora_args:
            cmd.extend(["--lora", la])

    # Prompt enhancement
    if enhance_prompt:
        cmd.append("--enhance-prompt")
        if mode == "i2v":
            cmd.extend(["--enhance-mode", "i2v"])

    # Launch subprocess — ensure ffmpeg is findable (Xcode strips PATH)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    path = env.get("PATH", "")
    for extra in ("/opt/homebrew/bin", "/usr/local/bin"):
        if extra not in path:
            path = f"{extra}:{path}"
    env["PATH"] = path

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=backend_dir,
        env=env,
    )

    # Parse stderr for progress
    subprocess_memory: dict[str, dict] = {}
    last_pct = 0.0
    last_step, last_total, last_stage = 0, 0, 1
    # Keep last N stderr lines for error diagnosis (readline consumes them)
    from collections import deque
    _stderr_tail: deque[str] = deque(maxlen=100)

    assert proc.stderr is not None
    while True:
        line_bytes = await proc.stderr.readline()
        if not line_bytes:
            break
        line = line_bytes.decode("utf-8", errors="replace").rstrip()
        if line:
            _stderr_tail.append(line)

        # PREVIEW frame
        m = _PREVIEW_RE.match(line)
        if m:
            fpath = m.group(1).strip()
            try:
                with open(fpath, "rb") as f:
                    b64_frame = base64.b64encode(f.read()).decode("ascii")
                os.unlink(fpath)
                if progress_callback:
                    r = progress_callback(
                        last_step, last_total, last_pct, b64_frame, status=None,
                    )
                    if asyncio.iscoroutine(r):
                        await r
            except Exception:
                log.debug("Failed to read preview frame: %s", fpath)
            continue

        # STAGE/STEP progress
        m = _STAGE_RE.match(line)
        if m:
            stage, step, total = int(m.group(1)), int(m.group(2)), int(m.group(3))
            pct = _compute_progress(stage, step, total)
            last_pct, last_step, last_total, last_stage = pct, step, total, stage
            if progress_callback:
                r = progress_callback(step, total, pct, None, status="Generating video")
                if asyncio.iscoroutine(r):
                    await r
            continue

        # MEMORY snapshot
        m = _MEMORY_RE.match(line)
        if m:
            label = m.group(1)
            subprocess_memory[label] = {
                "active_memory_gb": float(m.group(2)),
                "cache_memory_gb": float(m.group(3)),
                "peak_memory_gb": float(m.group(4)),
            }
            log.info("MEMORY[%s] active=%.1fGB cache=%.1fGB peak=%.1fGB",
                     label, float(m.group(2)), float(m.group(3)), float(m.group(4)))
            continue

        # STATUS message
        m = _STATUS_RE.match(line)
        if m:
            status_msg = m.group(1).strip()
            status_lower = status_msg.lower()
            for key, pct_val in _STATUS_PROGRESS.items():
                if key in status_lower:
                    last_pct = pct_val
                    break
            if progress_callback:
                r = progress_callback(last_step, last_total, last_pct, None, status=status_msg)
                if asyncio.iscoroutine(r):
                    await r
            continue

        # Other stderr lines -> log
        if line:
            log.debug("subprocess: %s", line[-200:])

    await proc.wait()

    if proc.returncode != 0:
        # Build error message from captured stderr tail (readline already consumed everything)
        error_tail = "\n".join(_stderr_tail)[-1000:]
        if proc.returncode == -6:
            raise RuntimeError(f"GPU out of memory (exit code -6). {error_tail}")
        raise RuntimeError(
            f"Generation subprocess failed (exit {proc.returncode}). {error_tail}"
        )

    return {
        "output_path": output_path,
        "subprocess_memory": subprocess_memory,
    }


# ---------------------------------------------------------------------------
# Prompt enhancement (subprocess)
# ---------------------------------------------------------------------------

async def run_prompt_enhance(
    prompt: str,
    is_i2v: bool = False,
    model_repo_id: str | None = None,
    venv_python: str | None = None,
) -> str:
    """Run prompt enhancement in a subprocess via Gemma.

    Returns the enhanced prompt string.
    """
    python_bin = venv_python or get_venv_python()
    model_repo, _ = get_model_repo(model_repo_id)
    backend_dir = str(Path(__file__).resolve().parent.parent)

    cmd = [
        python_bin, "-m", "engine.generate_v23",
        "--mode", "enhance",
        "--prompt", prompt,
        "--model-dir", model_repo,
        "--enhance-mode", "i2v" if is_i2v else "t2v",
        "--seed", "10",
    ]

    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    path = env.get("PATH", "")
    for extra in ("/opt/homebrew/bin", "/usr/local/bin"):
        if extra not in path:
            path = f"{extra}:{path}"
    env["PATH"] = path

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=backend_dir,
        env=env,
    )

    stdout_bytes, stderr_bytes = await proc.communicate()

    if proc.returncode != 0:
        error = stderr_bytes.decode("utf-8", errors="replace")[-500:]
        raise RuntimeError(f"Prompt enhancement failed (exit {proc.returncode}). {error}")

    return stdout_bytes.decode("utf-8").strip()
