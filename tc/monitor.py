"""Local progress and cooperative stopping; silence does not imply failure."""

from __future__ import annotations
from typing import Any

import json
from pathlib import Path
import time
from uuid import UUID

from tc.job_utils import write_json


def monitor(args: Any) -> None:
    """Read, watch or cooperatively stop one explicitly identified job."""
    identity = str(UUID(args.job_id))
    directory = Path(args.jobs_dir) / identity
    path = directory / "status.json"
    if not path.exists():
        raise ValueError("unknown job or status not yet written")
    status = json.loads(path.read_text())
    terminal = {"completed", "failed", "interrupted"}
    if args.command == "stop":
        if status["state"] not in terminal:
            write_json(directory / "stop.request", {"job_id": identity})
        print("Stop requested; workers check between batches." if status["state"] not in terminal else status["state"])
        return
    previous = None
    while True:
        current = path.read_text()
        if current != previous:
            print(current.strip(), flush=True)
            previous = current
        if args.command == "status" or json.loads(current)["state"] in terminal:
            return
        time.sleep(2)
