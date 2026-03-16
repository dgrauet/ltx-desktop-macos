"""Model download and disk management for LTX Desktop.

Handles model discovery (checking download status on disk), downloading
models via huggingface_hub, tracking download progress, and deleting
models to free disk space.
"""

from __future__ import annotations

import logging
import shutil
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# HuggingFace cache root
_HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

# Known model definitions: id -> metadata
_KNOWN_MODELS: list[dict[str, Any]] = [
    {
        "id": "ltx-2.3-distilled-q8",
        "name": "LTX-2.3 Distilled (int8)",
        "description": "Recommended. Best quality/size balance. 8-step distilled, int8 quantized.",
        "size_gb": 28.0,
        "model_type": "video_generator",
        "hf_repo": "dgrauet/ltx-2.3-mlx-distilled-q8",
        "check_path": _HF_CACHE / "models--dgrauet--ltx-2.3-mlx-distilled-q8",
        "check_file": "transformer.safetensors",
    },
    {
        "id": "ltx-2.3-distilled-q4",
        "name": "LTX-2.3 Distilled (int4)",
        "description": "Smaller and faster. Some quality loss. 8-step distilled, int4 quantized.",
        "size_gb": 15.0,
        "model_type": "video_generator",
        "hf_repo": "dgrauet/ltx-2.3-mlx-distilled-q4",
        "check_path": _HF_CACHE / "models--dgrauet--ltx-2.3-mlx-distilled-q4",
        "check_file": "transformer.safetensors",
    },
    {
        "id": "ltx-2.3-distilled-bf16",
        "name": "LTX-2.3 Distilled (bf16)",
        "description": "Full precision. Best quality, largest size. Requires 64GB+ RAM.",
        "size_gb": 42.0,
        "model_type": "video_generator",
        "hf_repo": "dgrauet/ltx-2.3-mlx-distilled",
        "check_path": _HF_CACHE / "models--dgrauet--ltx-2.3-mlx-distilled",
        "check_file": "transformer.safetensors",
    },
    {
        "id": "gemma-3-12b-it-4bit",
        "name": "Gemma 3 12B IT (4-bit)",
        "description": "Text encoder for video generation. Converts prompts to embeddings understood by LTX-2.3.",
        "size_gb": 6.0,
        "model_type": "text_encoder",
        "hf_repo": "mlx-community/gemma-3-12b-it-4bit",
        "check_file": "config.json",
    },
    {
        "id": "qwen3.5-2b-4bit",
        "name": "Qwen3.5-2B (4-bit)",
        "description": "Prompt enhancement model. Expands short prompts into detailed LTX-2.3 optimized descriptions.",
        "size_gb": 1.2,
        "model_type": "prompt_enhancer",
        "hf_repo": "mlx-community/Qwen3.5-2B-4bit",
        "check_file": "config.json",
    },
    {
        "id": "ltx-2.3-spatial-upscaler",
        "name": "LTX-2.3 Spatial Upscaler",
        "description": "Neural 2x upscaler for video frames. Used in two-stage pipeline for higher resolution output.",
        "size_gb": 1.0,
        "model_type": "upscaler",
        "hf_repo": "Lightricks/LTX-2.3",
        "hf_allow_patterns": ["ltx-2.3-spatial-upscaler-x2-1.0.safetensors"],
    },
]


def _dir_size_gb(path: Path) -> float:
    """Calculate directory size in GB."""
    if not path.exists():
        return 0.0
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024**3)


def _is_downloaded(model_def: dict[str, Any]) -> bool:
    """Check if a model is fully downloaded by verifying key files exist in cache."""
    try:
        from huggingface_hub import try_to_load_from_cache

        repo = model_def["hf_repo"]

        # For partial downloads (upscaler), check allow_patterns files
        allow_patterns = model_def.get("hf_allow_patterns")
        if allow_patterns:
            for fn in allow_patterns:
                cached = try_to_load_from_cache(repo, fn)
                if not cached or not isinstance(cached, str):
                    return False
            return True

        # For full models, check the key file (e.g. transformer.safetensors)
        check_file = model_def.get("check_file", "config.json")
        cached = try_to_load_from_cache(repo, check_file)
        return cached is not None and isinstance(cached, str)

    except Exception:
        return False


class ModelDownloadManager:
    """Manages model listing, downloading, and deletion."""

    def __init__(self) -> None:
        self._downloads: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def list_models(self) -> list[dict[str, Any]]:
        """Return list of all known models with download status.

        Returns:
            List of dicts with id, name, description, size_gb, model_type,
            downloaded, hf_repo.
        """
        result = []
        for m in _KNOWN_MODELS:
            downloaded = _is_downloaded(m)
            actual_size = m["size_gb"]
            if downloaded:
                cache_dir = _HF_CACHE / ("models--" + m["hf_repo"].replace("/", "--"))
                disk_size = _dir_size_gb(cache_dir)
                if disk_size > 0.1:
                    actual_size = round(disk_size, 2)
            result.append(
                {
                    "id": m["id"],
                    "name": m["name"],
                    "description": m["description"],
                    "size_gb": actual_size,
                    "model_type": m["model_type"],
                    "downloaded": downloaded,
                    "hf_repo": m["hf_repo"],
                }
            )
        return result

    def get_model(self, model_id: str) -> dict[str, Any] | None:
        """Get model info by ID, or None if unknown.

        Args:
            model_id: The model identifier.

        Returns:
            Model info dict or None.
        """
        for m in _KNOWN_MODELS:
            if m["id"] == model_id:
                return {
                    "id": m["id"],
                    "name": m["name"],
                    "description": m["description"],
                    "size_gb": m["size_gb"],
                    "model_type": m["model_type"],
                    "downloaded": _is_downloaded(m),
                    "hf_repo": m["hf_repo"],
                }
        return None

    def start_download(self, download_id: str, model_id: str) -> None:
        """Initialize download tracking entry.

        Args:
            download_id: Unique download tracking ID.
            model_id: The model to download.
        """
        with self._lock:
            self._downloads[download_id] = {
                "download_id": download_id,
                "model_id": model_id,
                "status": "pending",
                "progress": 0.0,
                "error": None,
            }

    def download_model(self, download_id: str, model_id: str) -> None:
        """Download a model from HuggingFace Hub. Blocking call.

        Updates download progress in self._downloads. Uses huggingface_hub
        snapshot_download with standard HF cache.

        Args:
            download_id: Tracking ID for progress updates.
            model_id: The model to download.

        Raises:
            ValueError: If model_id is unknown.
            RuntimeError: If download fails.
        """
        model_def = None
        for m in _KNOWN_MODELS:
            if m["id"] == model_id:
                model_def = m
                break
        if model_def is None:
            raise ValueError(f"Unknown model: {model_id}")

        with self._lock:
            self._downloads[download_id]["status"] = "downloading"
            self._downloads[download_id]["progress"] = 0.05

        try:
            from huggingface_hub import snapshot_download

            hf_repo = model_def["hf_repo"]

            with self._lock:
                self._downloads[download_id]["progress"] = 0.1

            allow_patterns = model_def.get("hf_allow_patterns")
            # force_download=False (default) is fine — HF Hub will skip already-cached files.
            # The key is that snapshot_download fetches ALL files (including large safetensors).
            snapshot_download(
                repo_id=hf_repo,
                **({"allow_patterns": allow_patterns} if allow_patterns else {}),
            )

            # Verify the download is complete (safetensors files exist in snapshot)
            from huggingface_hub import try_to_load_from_cache
            verify_file = allow_patterns[0] if allow_patterns else "config.json"
            cached = try_to_load_from_cache(hf_repo, verify_file)
            if not cached or not isinstance(cached, str):
                raise RuntimeError(
                    f"Download incomplete: {verify_file} not found in cache for {hf_repo}"
                )

            with self._lock:
                self._downloads[download_id]["status"] = "completed"
                self._downloads[download_id]["progress"] = 1.0

            log.info("Download %s completed: %s", download_id, model_id)

        except Exception as e:
            log.error("Download %s failed: %s", download_id, e)
            with self._lock:
                self._downloads[download_id]["status"] = "failed"
                self._downloads[download_id]["error"] = str(e)
            raise

    def fail_download(self, download_id: str, error: str) -> None:
        """Mark a download as failed.

        Args:
            download_id: The download tracking ID.
            error: Error message.
        """
        with self._lock:
            if download_id in self._downloads:
                self._downloads[download_id]["status"] = "failed"
                self._downloads[download_id]["error"] = error

    def get_download_status(self, download_id: str) -> dict[str, Any] | None:
        """Get current download progress.

        Args:
            download_id: The download tracking ID.

        Returns:
            Download status dict or None if unknown.
        """
        with self._lock:
            return self._downloads.get(download_id)

    def delete_model(self, model_id: str) -> float:
        """Delete a downloaded model from disk.

        Args:
            model_id: The model to delete.

        Returns:
            Amount of disk space freed in GB.

        Raises:
            ValueError: If model_id is unknown.
            FileNotFoundError: If model is not downloaded.
        """
        model_def = None
        for m in _KNOWN_MODELS:
            if m["id"] == model_id:
                model_def = m
                break
        if model_def is None:
            raise ValueError(f"Unknown model: {model_id}")

        hf_repo = model_def["hf_repo"]
        cache_dir_name = "models--" + hf_repo.replace("/", "--")
        check_path = _HF_CACHE / cache_dir_name
        if not check_path.exists():
            raise FileNotFoundError(f"Model not found on disk: {model_id}")

        freed = _dir_size_gb(check_path)
        shutil.rmtree(check_path)
        log.info("Deleted model %s at %s, freed %.2f GB", model_id, check_path, freed)
        return freed
