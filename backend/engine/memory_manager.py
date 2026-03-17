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


def build_memory_stats_from_subprocess(subprocess_memory: dict[str, dict[str, float]]) -> dict:
    """Build a memory stats dict from subprocess-reported memory data.

    Uses the 'final' snapshot if available, otherwise the last reported snapshot.
    Falls back to get_memory_stats() if no subprocess data is available.

    Args:
        subprocess_memory: Dict of label -> {active_memory_gb, cache_memory_gb, peak_memory_gb}
            as reported by the generation subprocess.

    Returns:
        Dictionary in the same format as get_memory_stats().
    """
    if not subprocess_memory:
        return get_memory_stats()

    # Prefer 'final' snapshot, fall back to last reported
    snapshot = subprocess_memory.get("final")
    if snapshot is None:
        # Use the last entry (dicts are ordered in Python 3.7+)
        snapshot = list(subprocess_memory.values())[-1]

    return {
        "active_memory_gb": round(snapshot.get("active_memory_gb", 0.0), 3),
        "cache_memory_gb": round(snapshot.get("cache_memory_gb", 0.0), 3),
        "peak_memory_gb": round(snapshot.get("peak_memory_gb", 0.0), 3),
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


class MemoryPressureMonitor:
    """Monitors memory pressure and takes automated actions.

    Actions:
    - cache > 2x active → aggressive_cleanup()
    - available < 2GB → pause job queue
    - available > 4GB after pause → resume job queue
    """

    def __init__(self) -> None:
        self.auto_pause_enabled: bool = True
        self.auto_cleanup_enabled: bool = True
        self._paused_by_pressure: bool = False
        self._last_actions: list[str] = []

    def check_pressure(self, job_queue: object | None = None) -> dict:
        """Check memory pressure and take automated actions.

        Args:
            job_queue: JobQueue instance with pause()/resume()/is_paused.

        Returns:
            Dictionary with pressure level, actions taken, and current state.
        """
        stats = get_memory_stats()
        actions: list[str] = []
        pressure_level = "normal"

        active = stats["active_memory_gb"]
        cache = stats["cache_memory_gb"]
        available = stats["system_available_gb"]

        # Check cache pressure
        if self.auto_cleanup_enabled and active > 0 and cache > 2 * active:
            aggressive_cleanup()
            actions.append("cache_cleanup")
            pressure_level = "warning"
            log.info(
                "Memory pressure: cache (%.2f GB) > 2x active (%.2f GB) — cleaned up",
                cache, active,
            )

        # Check low available memory → pause queue
        if self.auto_pause_enabled and available < 2.0 and job_queue is not None:
            if not self._paused_by_pressure:
                _pause_queue(job_queue)
                self._paused_by_pressure = True
                actions.append("queue_paused")
                log.warning(
                    "Memory pressure: available %.2f GB < 2 GB — pausing queue",
                    available,
                )
            pressure_level = "critical"

        # Check recovery → resume queue
        if self._paused_by_pressure and available > 4.0 and job_queue is not None:
            _resume_queue(job_queue)
            self._paused_by_pressure = False
            actions.append("queue_resumed")
            log.info(
                "Memory pressure recovered: available %.2f GB > 4 GB — resuming queue",
                available,
            )
            if pressure_level == "normal":
                pressure_level = "normal"

        # Set pressure level based on thresholds even if no action taken
        if pressure_level == "normal":
            if available < 4.0:
                pressure_level = "warning"
            if active > 0 and cache > 2 * active:
                pressure_level = "warning"

        self._last_actions = actions

        return {
            "pressure_level": pressure_level,
            "actions_taken": actions,
            "paused_by_pressure": self._paused_by_pressure,
            "auto_pause_enabled": self.auto_pause_enabled,
            "auto_cleanup_enabled": self.auto_cleanup_enabled,
            "memory": stats,
        }

    def get_state(self) -> dict:
        """Return current monitor state without taking actions."""
        return {
            "paused_by_pressure": self._paused_by_pressure,
            "auto_pause_enabled": self.auto_pause_enabled,
            "auto_cleanup_enabled": self.auto_cleanup_enabled,
            "last_actions": self._last_actions,
        }

    def manual_resume(self, job_queue: object | None = None) -> bool:
        """Manually resume a queue paused by memory pressure.

        Returns:
            True if the queue was resumed.
        """
        if self._paused_by_pressure and job_queue is not None:
            _resume_queue(job_queue)
            self._paused_by_pressure = False
            log.info("Memory pressure pause manually overridden — queue resumed")
            return True
        return False


def _pause_queue(job_queue: object) -> None:
    """Pause a job queue (duck-typed)."""
    if hasattr(job_queue, "pause"):
        job_queue.pause()  # type: ignore[attr-defined]


def _resume_queue(job_queue: object) -> None:
    """Resume a job queue (duck-typed)."""
    if hasattr(job_queue, "resume"):
        job_queue.resume()  # type: ignore[attr-defined]


# --- Singleton monitor instance ---
memory_pressure_monitor = MemoryPressureMonitor()


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
