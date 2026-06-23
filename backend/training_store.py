"""Disk-backed training-run persistence (one run.json per run dir)."""
from __future__ import annotations
import json
import shutil
from pathlib import Path

TRAINING_DIR = Path.home() / ".ltx-desktop" / "training"
RUNS_DIR = TRAINING_DIR / "runs"
_VALID = {"pending", "preprocessing", "preflight", "training", "completed", "failed", "cancelled"}


def run_dir(run_id: str) -> Path:
    return RUNS_DIR / run_id


def _read(run_id: str) -> dict | None:
    p = run_dir(run_id) / "run.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _write(run_id: str, data: dict) -> None:
    d = run_dir(run_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.json").write_text(json.dumps(data, indent=2, default=str))


def create_run(run_id: str, *, dataset_id: str, config_path: str, created_at: str) -> dict:
    """Create a new training run record with status='pending'.

    Args:
        run_id: Unique identifier for the run.
        dataset_id: Dataset used for training.
        config_path: Path to the training config YAML.
        created_at: ISO-8601 creation timestamp.

    Returns:
        The newly created run dict.
    """
    data = {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "config_path": config_path,
        "created_at": created_at,
        "status": "pending",
    }
    _write(run_id, data)
    return data


def update_run(run_id: str, **fields) -> dict:
    """Merge fields into an existing run record.

    Args:
        run_id: Run to update.
        **fields: Fields to merge (status, peak_mem_gb, lora_path, etc.).

    Returns:
        The updated run dict.

    Raises:
        ValueError: If ``status`` is not in the allowed set.
    """
    data = _read(run_id) or {"run_id": run_id}
    if "status" in fields and fields["status"] not in _VALID:
        raise ValueError(f"invalid status {fields['status']!r}")
    data.update(fields)
    _write(run_id, data)
    return data


def get_run(run_id: str) -> dict | None:
    """Return the run dict for run_id, or None if not found.

    Args:
        run_id: Run to look up.

    Returns:
        Run dict or None.
    """
    return _read(run_id)


def list_runs() -> list[dict]:
    """List all training runs, newest first. Corrupt run.json entries are skipped.

    Returns:
        List of run dicts sorted by created_at descending.
    """
    if not RUNS_DIR.exists():
        return []
    out = []
    for d in RUNS_DIR.iterdir():
        if d.is_dir():
            data = _read(d.name)
            if data:
                out.append(data)
    return sorted(out, key=lambda r: r.get("created_at", ""), reverse=True)


def delete_run(run_id: str) -> bool:
    """Remove a training run directory.

    Args:
        run_id: Run to delete.

    Returns:
        True if the run existed and was deleted, False otherwise.
    """
    d = run_dir(run_id)
    if d.exists():
        shutil.rmtree(d)
        return True
    return False


def disk_usage_bytes(run_id: str) -> int:
    """Return total bytes used by a run's directory.

    Args:
        run_id: Run to measure.

    Returns:
        Sum of file sizes in bytes, or 0 if the directory does not exist.
    """
    d = run_dir(run_id)
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) if d.exists() else 0
