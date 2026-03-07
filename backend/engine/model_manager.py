"""Model loading and lifecycle management.

Sprint 1 stub — provides the interface that memory_manager and pipelines
depend on. Real MLX model loading will be implemented in Sprint 2+.
"""

from __future__ import annotations

import logging

from engine.memory_manager import aggressive_cleanup

log = logging.getLogger(__name__)


class ModelManager:
    """Manages MLX model loading, unloading, and lifecycle."""

    def __init__(self) -> None:
        self._loaded_model_id: str | None = None
        self._model: object | None = None

    def load_model(self, model_id: str) -> None:
        """Load a model by ID.

        Args:
            model_id: HuggingFace model ID or local path.
        """
        log.info("Loading model: %s", model_id)
        self._loaded_model_id = model_id
        # Sprint 1 stub: no actual model loading
        self._model = {"id": model_id, "stub": True}
        log.info("Model loaded (stub): %s", model_id)

    def unload_all(self) -> None:
        """Unload all models and free memory."""
        if self._loaded_model_id:
            log.info("Unloading model: %s", self._loaded_model_id)
        self._model = None
        self._loaded_model_id = None
        aggressive_cleanup()
        log.info("All models unloaded")

    def is_loaded(self) -> bool:
        """Check if any model is currently loaded."""
        return self._loaded_model_id is not None

    def get_model_id(self) -> str | None:
        """Get the currently loaded model ID, or None."""
        return self._loaded_model_id

    def get_model(self) -> object | None:
        """Get the loaded model object."""
        return self._model
