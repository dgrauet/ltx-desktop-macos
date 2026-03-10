"""Priority job queue for batch generation management.

Manages a FIFO-within-priority queue where only one GPU job runs at a time.
Preview jobs get high priority (fast, cut the line), normal generations get
normal priority, and batch submissions get low priority.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Coroutine

log = logging.getLogger(__name__)


class Priority(IntEnum):
    """Job priority levels. Lower value = higher priority."""

    HIGH = 0  # preview jobs — fast, cut the line
    NORMAL = 1  # t2v, i2v — standard generation
    LOW = 2  # batch queue submissions


class JobState:
    """Job lifecycle states."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class QueuedJob:
    """A job waiting in or running through the queue."""

    job_id: str
    job_type: str  # "t2v", "i2v", "preview", "retake", "extend"
    priority: Priority
    prompt: str  # For display in queue UI
    coroutine_factory: Callable[[], Coroutine[Any, Any, None]]
    state: str = JobState.QUEUED
    submitted_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    task: asyncio.Task[None] | None = None


class JobQueue:
    """Priority-based generation job queue.

    Only one job runs at a time (GPU can't parallelize on 32GB).
    Jobs are FIFO within the same priority level.
    Preview jobs auto-cancel any previous queued preview job.

    Usage:
        queue = JobQueue()
        await queue.start()
        position = await queue.submit("abc", "t2v", Priority.NORMAL, "a cat", coro_factory)
        ...
        await queue.stop()
    """

    def __init__(self) -> None:
        self._queues: dict[Priority, deque[QueuedJob]] = {
            p: deque() for p in Priority
        }
        self._running_job: QueuedJob | None = None
        self._all_jobs: dict[str, QueuedJob] = {}
        self._lock = asyncio.Lock()
        self._process_event = asyncio.Event()
        self._processor_task: asyncio.Task[None] | None = None
        # ETA tracking: job_type -> list of recent durations (up to 5)
        self._duration_history: dict[str, list[float]] = {}
        # Cancellation callback: job_id -> callable that cancels the running subprocess
        self._cancel_callbacks: dict[str, Callable[[], None]] = {}

    async def start(self) -> None:
        """Start the background queue processor."""
        if self._processor_task is None:
            self._processor_task = asyncio.create_task(self._process_loop())
            log.info("Job queue processor started")

    async def stop(self) -> None:
        """Stop the queue processor and cancel any running job."""
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
            self._processor_task = None
            log.info("Job queue processor stopped")

    async def submit(
        self,
        job_id: str,
        job_type: str,
        priority: Priority,
        prompt: str,
        coroutine_factory: Callable[[], Coroutine[Any, Any, None]],
    ) -> int:
        """Submit a job to the queue.

        Args:
            job_id: Unique job identifier.
            job_type: Type of generation ("t2v", "i2v", "preview", etc.).
            priority: Job priority level.
            prompt: Text prompt (for queue display).
            coroutine_factory: A zero-arg callable that returns the coroutine
                to execute. Must be a factory (not a coroutine directly) so we
                can create it when the job actually starts.

        Returns:
            Position in queue (1-based). 0 means it will run immediately.
        """
        async with self._lock:
            # Preview jobs auto-cancel any previous queued preview
            if job_type == "preview":
                self._cancel_queued_previews()

            job = QueuedJob(
                job_id=job_id,
                job_type=job_type,
                priority=priority,
                prompt=prompt,
                coroutine_factory=coroutine_factory,
            )
            self._all_jobs[job_id] = job
            self._queues[priority].append(job)

            position = self._get_position(job_id)
            log.info(
                "Job %s (%s, %s) queued at position %d",
                job_id, job_type, priority.name, position,
            )

        # Wake up the processor
        self._process_event.set()
        return position

    async def cancel(self, job_id: str) -> bool:
        """Cancel a queued or running job.

        Args:
            job_id: The job to cancel.

        Returns:
            True if the job was found and cancelled.
        """
        async with self._lock:
            job = self._all_jobs.get(job_id)
            if job is None:
                return False

            if job.state == JobState.QUEUED:
                # Remove from priority queue
                job.state = JobState.CANCELLED
                try:
                    self._queues[job.priority].remove(job)
                except ValueError:
                    pass
                log.info("Cancelled queued job %s", job_id)
                return True

            if job.state == JobState.RUNNING:
                job.state = JobState.CANCELLED
                # Cancel the asyncio task
                if job.task is not None:
                    job.task.cancel()
                # Call the subprocess cancellation callback if registered
                cancel_cb = self._cancel_callbacks.pop(job_id, None)
                if cancel_cb is not None:
                    try:
                        cancel_cb()
                    except Exception as e:
                        log.warning("Cancel callback for %s failed: %s", job_id, e)
                log.info("Cancelled running job %s", job_id)
                return True

        return False

    def register_cancel_callback(self, job_id: str, callback: Callable[[], None]) -> None:
        """Register a callback to kill the subprocess for a running job.

        Args:
            job_id: The running job's ID.
            callback: Called when cancellation is requested. Should terminate
                the subprocess (e.g., proc.terminate()).
        """
        self._cancel_callbacks[job_id] = callback

    async def reorder(self, job_id: str, new_priority: Priority) -> bool:
        """Change a queued job's priority.

        Args:
            job_id: The job to reorder.
            new_priority: New priority level.

        Returns:
            True if the job was reordered.
        """
        async with self._lock:
            job = self._all_jobs.get(job_id)
            if job is None or job.state != JobState.QUEUED:
                return False

            old_priority = job.priority
            if old_priority == new_priority:
                return True

            # Remove from old queue, add to new
            try:
                self._queues[old_priority].remove(job)
            except ValueError:
                return False

            job.priority = new_priority
            self._queues[new_priority].append(job)
            log.info(
                "Reordered job %s from %s to %s",
                job_id, old_priority.name, new_priority.name,
            )
            return True

    def get_queue_state(self) -> list[dict[str, Any]]:
        """Return the full queue state for the API.

        Returns:
            List of job dicts with id, type, priority, state, position, prompt,
            submitted_at, eta_seconds.
        """
        result: list[dict[str, Any]] = []

        # Running job first
        if self._running_job and self._running_job.state == JobState.RUNNING:
            j = self._running_job
            result.append(self._job_to_dict(j, position=0))

        # Then queued jobs in priority order
        pos = 1
        for priority in Priority:
            for job in self._queues[priority]:
                if job.state == JobState.QUEUED:
                    result.append(self._job_to_dict(job, position=pos))
                    pos += 1

        return result

    def get_queue_length(self) -> int:
        """Return the number of queued (not yet running) jobs."""
        return sum(
            1
            for pq in self._queues.values()
            for j in pq
            if j.state == JobState.QUEUED
        )

    def get_job(self, job_id: str) -> QueuedJob | None:
        """Get a job by ID."""
        return self._all_jobs.get(job_id)

    def record_duration(self, job_type: str, duration: float) -> None:
        """Record a completed job's duration for ETA estimation.

        Args:
            job_type: The type of job (e.g., "t2v", "preview").
            duration: Duration in seconds.
        """
        history = self._duration_history.setdefault(job_type, [])
        history.append(duration)
        # Keep only the last 5
        if len(history) > 5:
            self._duration_history[job_type] = history[-5:]

    def estimate_eta(self, job_type: str) -> float | None:
        """Estimate time for a job type based on recent history.

        Returns:
            Estimated seconds, or None if no history.
        """
        history = self._duration_history.get(job_type)
        if not history:
            return None
        return sum(history) / len(history)

    # --- Internal ---

    def _cancel_queued_previews(self) -> None:
        """Cancel all queued preview jobs (only latest preview matters)."""
        q = self._queues[Priority.HIGH]
        to_cancel = [j for j in q if j.job_type == "preview" and j.state == JobState.QUEUED]
        for j in to_cancel:
            j.state = JobState.CANCELLED
            q.remove(j)
            log.info("Auto-cancelled previous preview job %s", j.job_id)

    def _get_position(self, job_id: str) -> int:
        """Get the position of a job in the queue (1-based). 0 if it would run next."""
        pos = 0
        if self._running_job and self._running_job.state == JobState.RUNNING:
            pos += 1  # Account for the running job
        for priority in Priority:
            for job in self._queues[priority]:
                if job.state == JobState.QUEUED:
                    if job.job_id == job_id:
                        return pos
                    pos += 1
        return pos

    def _job_to_dict(self, job: QueuedJob, position: int) -> dict[str, Any]:
        """Convert a QueuedJob to a dict for the API."""
        eta = self.estimate_eta(job.job_type)
        # If queued, multiply ETA by position (rough approximation)
        if position > 0 and eta is not None:
            eta = eta * position

        return {
            "job_id": job.job_id,
            "job_type": job.job_type,
            "priority": job.priority.name.lower(),
            "state": job.state,
            "position": position,
            "prompt": job.prompt[:100],  # Truncate for display
            "submitted_at": job.submitted_at,
            "eta_seconds": round(eta, 1) if eta is not None else None,
        }

    def _next_job(self) -> QueuedJob | None:
        """Pop the next job to run (highest priority, FIFO within level)."""
        for priority in Priority:
            q = self._queues[priority]
            while q:
                job = q[0]
                if job.state == JobState.QUEUED:
                    return q.popleft()
                # Skip cancelled jobs
                q.popleft()
        return None

    async def _process_loop(self) -> None:
        """Background loop that processes queued jobs one at a time."""
        while True:
            await self._process_event.wait()
            self._process_event.clear()

            while True:
                async with self._lock:
                    if self._running_job and self._running_job.state == JobState.RUNNING:
                        # Already running a job, wait for it to finish
                        break
                    job = self._next_job()

                if job is None:
                    break  # Queue empty

                if job.state == JobState.CANCELLED:
                    continue

                # Start the job
                async with self._lock:
                    job.state = JobState.RUNNING
                    job.started_at = time.time()
                    self._running_job = job

                log.info("Starting job %s (%s, %s)", job.job_id, job.job_type, job.priority.name)

                try:
                    coro = job.coroutine_factory()
                    task = asyncio.create_task(coro)
                    job.task = task
                    await task
                except asyncio.CancelledError:
                    log.info("Job %s was cancelled", job.job_id)
                    job.state = JobState.CANCELLED
                except Exception as e:
                    log.error("Job %s failed in queue processor: %s", job.job_id, e)
                    # The _run_* functions handle their own error states,
                    # so we only set FAILED if it wasn't already set
                    if job.state == JobState.RUNNING:
                        job.state = JobState.FAILED
                finally:
                    job.completed_at = time.time()
                    self._cancel_callbacks.pop(job.job_id, None)

                    # Record duration for ETA
                    if job.started_at and job.completed_at and job.state != JobState.CANCELLED:
                        duration = job.completed_at - job.started_at
                        self.record_duration(job.job_type, duration)

                    async with self._lock:
                        self._running_job = None

                    log.info(
                        "Job %s finished with state %s",
                        job.job_id, job.state,
                    )

            # After processing all ready jobs, go back to waiting
