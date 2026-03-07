"""Metal memory management for MLX inference.

Provides aggressive cleanup, memory monitoring, and periodic model reload
to prevent Metal buffer fragmentation across repeated generations.
"""

from __future__ import annotations

import gc
import logging
import platform
import subprocess

log = logging.getLogger(__name__)

# --- MLX availability (graceful fallback for non-Apple-Silicon) ---

try:
    import mlx.core as mx

    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False

# --- Generation counter for periodic reload ---

_generation_count: int = 0
MAX_GENERATIONS_BEFORE_RELOAD: int = 5


def aggressive_cleanup() -> None:
    """Force-free Metal memory between pipeline stages.

    Must be called between every major stage and between every generation.
    Uses gc.collect(), mx.metal.clear_cache(), and a GPU synchronization barrier.
    """
    gc.collect()
    if _MLX_AVAILABLE:
        mx.clear_cache()
        # GPU synchronization barrier — forces all pending ops to complete
        mx.synchronize()
    log.debug("aggressive_cleanup complete")


def get_memory_stats() -> dict:
    """Return current Metal memory statistics.

    Returns:
        Dictionary with active, cache, peak, and system available memory in GB,
        plus generation count and next reload countdown.
    """
    if _MLX_AVAILABLE:
        active = mx.get_active_memory() / (1024**3)
        cache = mx.get_cache_memory() / (1024**3)
        peak = mx.get_peak_memory() / (1024**3)
    else:
        active = cache = peak = 0.0

    return {
        "active_memory_gb": round(active, 3),
        "cache_memory_gb": round(cache, 3),
        "peak_memory_gb": round(peak, 3),
        "system_available_gb": round(_get_system_available_memory_gb(), 3),
        "generation_count_since_reload": _generation_count,
        "next_reload_in": MAX_GENERATIONS_BEFORE_RELOAD - _generation_count,
    }


def reset_peak_memory() -> None:
    """Reset the peak memory counter for per-generation tracking."""
    if _MLX_AVAILABLE:
        mx.reset_peak_memory()


def increment_generation_count() -> None:
    """Increment the generation counter. Called after each completed generation."""
    global _generation_count
    _generation_count += 1
    log.info(
        "Generation %d/%d before next model reload",
        _generation_count,
        MAX_GENERATIONS_BEFORE_RELOAD,
    )


def periodic_reload_check(model_manager: object) -> bool:
    """Check if a periodic model reload is needed to reclaim fragmented Metal memory.

    Args:
        model_manager: ModelManager instance with unload_all() and reload() methods.

    Returns:
        True if a reload was performed.
    """
    global _generation_count
    if _generation_count >= MAX_GENERATIONS_BEFORE_RELOAD:
        log.info("Periodic model reload to reclaim fragmented Metal memory")
        model_manager.unload_all()  # type: ignore[attr-defined]
        aggressive_cleanup()
        _generation_count = 0
        return True
    return False


def _get_system_available_memory_gb() -> float:
    """Get available system memory in GB (macOS-specific)."""
    if platform.system() != "Darwin":
        return 0.0
    try:
        # Use vm_stat for approximate available memory
        vm = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, check=True
        )
        free_pages = 0
        page_size = 16384  # Default on Apple Silicon
        for line in vm.stdout.splitlines():
            if "page size" in line.lower():
                page_size = int("".join(c for c in line if c.isdigit()) or "16384")
            if "Pages free" in line or "Pages speculative" in line:
                val = line.split(":")[1].strip().rstrip(".")
                free_pages += int(val)
        return (free_pages * page_size) / (1024**3)
    except Exception:
        log.debug("Failed to read system memory, returning 0")
        return 0.0
