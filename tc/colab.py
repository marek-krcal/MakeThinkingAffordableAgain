"""Explicit-session Colab CLI calls; no implicit default-runtime selection."""

from __future__ import annotations
from typing import Any

from contextlib import contextmanager
import fcntl
from pathlib import Path
import subprocess
from uuid import UUID


@contextmanager
def allocation_lock(directory: Path) -> Any:
    """Serialize allocation attempts within one local session directory."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "allocation.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def call(state: str | Path, *arguments: Any, executable: str = "colab", runner: Any=subprocess.run) -> Any:
    """Invoke Colab with an explicit authentication and session state directory."""
    state = Path(state).resolve()
    state.mkdir(parents=True, exist_ok=True)
    return runner([executable, "--auth", "adc", "--config", str(state / "sessions.json"), *arguments],
                  check=True, capture_output=True, text=True)


def session_name(job_id: str) -> str:
    """Derive a stable Colab session name from a job UUID."""
    return "tc-" + str(UUID(job_id))


def create(state: str | Path, job_id: str, *, gpu: str | None = None, runner: Any=subprocess.run) -> Any:
    """Allocate one named runtime without retrying ambiguous responses."""
    options = ["new", "--session", session_name(job_id)]
    if gpu:
        if gpu not in ("L4", "A100", "T4", "H100"):
            raise ValueError("unsupported GPU")
        options += ["--gpu", gpu]
    with allocation_lock(state):
        # No retries: reconcile ambiguous responses through sessions before another allocation.
        return call(state, *options, runner=runner)


def stop(state: str | Path, job_id: str, *, runner: Any=subprocess.run) -> Any:
    """Release only the runtime named by this job UUID."""
    return call(state, "stop", "--session", session_name(job_id), runner=runner)
