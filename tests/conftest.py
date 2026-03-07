"""Test fixtures for LTX Desktop macOS backend tests."""

from __future__ import annotations

import subprocess
import sys
import time

import httpx
import pytest

BACKEND_URL = "http://127.0.0.1:8000"


@pytest.fixture(scope="session")
def backend_process():
    """Start the backend server for testing and tear it down after."""
    # Check if already running
    try:
        r = httpx.get(f"{BACKEND_URL}/api/v1/system/health", timeout=2)
        if r.status_code == 200:
            yield BACKEND_URL
            return
    except Exception:
        pass

    # Start the backend
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "main:app", "--host", "127.0.0.1", "--port", "8000",
        ],
        cwd=str(
            __import__("pathlib").Path(__file__).parent.parent / "backend"
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for it to be ready
    for _ in range(60):
        try:
            r = httpx.get(f"{BACKEND_URL}/api/v1/system/health", timeout=2)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.5)
    else:
        proc.kill()
        pytest.fail("Backend failed to start within 30 seconds")

    yield BACKEND_URL

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="session")
def client(backend_process):
    """HTTP client for API tests."""
    return httpx.Client(base_url=backend_process, timeout=600)
