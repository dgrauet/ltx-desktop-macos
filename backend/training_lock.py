"""Mutual-exclusion primitive for training ↔ generation concurrency guard.

Training and generation both contend for the same GPU/unified-memory budget.
``ExclusionLock`` ensures that at most one holder occupies the lock at a time;
the same holder may re-acquire without deadlocking (idempotent acquire).
"""

from __future__ import annotations

import threading


class ExclusionLock:
    """A named, non-reentrant exclusion lock for mutually exclusive job types.

    Semantics:
    - ``try_acquire(holder)`` returns ``True`` if the lock is free *or* already
      held by the same holder, and ``False`` if held by a different holder.
    - ``release(holder)`` releases the lock only when *holder* matches the
      current holder; mismatched releases are silently ignored.
    - All state mutations are protected by a ``threading.Lock``.
    """

    def __init__(self) -> None:
        self._mutex: threading.Lock = threading.Lock()
        self._holder: str | None = None

    def try_acquire(self, holder: str) -> bool:
        """Attempt to acquire the lock for *holder*.

        Args:
            holder: Identifier for the caller (e.g. ``"training"`` or
                ``"generation"``).

        Returns:
            ``True`` if the lock was acquired or is already held by *holder*;
            ``False`` if the lock is held by a different holder.
        """
        with self._mutex:
            if self._holder is None or self._holder == holder:
                self._holder = holder
                return True
            return False

    def release(self, holder: str) -> None:
        """Release the lock if *holder* is the current holder.

        Args:
            holder: Identifier for the caller.  If this does not match the
                current holder the call is a no-op.
        """
        with self._mutex:
            if self._holder == holder:
                self._holder = None

    def current_holder(self) -> str | None:
        """Return the identifier of the current holder, or ``None`` if free."""
        with self._mutex:
            return self._holder

    def is_held(self) -> bool:
        """Return ``True`` if the lock is currently held by any holder."""
        with self._mutex:
            return self._holder is not None


# Module-level singleton used by generation endpoints and training start.
training_lock = ExclusionLock()
