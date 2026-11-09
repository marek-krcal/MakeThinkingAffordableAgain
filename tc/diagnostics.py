"""Small process diagnostics; copy this directory off the runtime regularly."""

from pathlib import Path
from typing import Any
import json
import os
import signal
import subprocess
import sys
import time


def event(directory: Path, **fields: Any) -> None:
    """Append a timestamped process diagnostic event."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps({'time': time.time(), **fields}) + '\n')
        stream.flush()


def snapshot(pid: int) -> dict[str, Any]:
    """Capture process, memory, OOM and GPU state without failing the worker."""
    result: dict[str, Any] = {'pid': pid}
    for name, path in {'process': f'/proc/{pid}/status', 'memory': '/proc/meminfo',
                       'oom': '/sys/fs/cgroup/memory.events'}.items():
        try:
            result[name] = Path(path).read_text()
        except OSError as error:
            result[name] = type(error).__name__
    try:
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu',
                              '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
        result['gpu'] = gpu.stdout.strip()
        result['gpu_returncode'] = gpu.returncode
    except (OSError, subprocess.TimeoutExpired) as error:
        result['gpu_error'] = type(error).__name__
    return result


def observe(pid: int, directory: Path, interval: float = 30) -> None:
    """Attach without restarting a running worker; its exit code is unavailable."""
    directory.mkdir(parents=True)
    event(directory, kind='attached', pid=pid)
    while True:
        state = snapshot(pid)
        event(directory, kind='sample', **state)
        if state['process'] == 'FileNotFoundError' or '\nState:\tZ' in state['process']:
            event(directory, kind='process_disappeared', pid=pid, exit_code=None)
            (directory / 'terminal.json').write_text(json.dumps({'pid': pid, 'exit_code': None}))
            return
        time.sleep(interval)


def supervise(command: list[str], directory: Path, interval: float = 30) -> int:
    """Run a child in fresh diagnostics storage and preserve its final exit status."""
    directory.mkdir(parents=True)
    with (directory / 'console.log').open('ab', buffering=0) as console:
        process = subprocess.Popen(command, stdout=console, stderr=subprocess.STDOUT,
                                   env={**os.environ, "PYTHONFAULTHANDLER": "1", "PYTHONUNBUFFERED": "1"})
        event(directory, kind='started', pid=process.pid)
        while process.poll() is None:
            event(directory, kind='sample', **snapshot(process.pid))
            try:
                process.wait(timeout=interval)
            except subprocess.TimeoutExpired:
                pass
        code = process.returncode
        event(directory, kind='exited', pid=process.pid, exit_code=code,
              signal=signal.Signals(-code).name if code < 0 else None)
        (directory / 'terminal.json').write_text(json.dumps({'pid': process.pid, 'exit_code': code}))
        return code


if __name__ == '__main__':
    raise SystemExit(supervise(sys.argv[2:], Path(sys.argv[1])))
