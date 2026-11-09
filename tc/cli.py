"""Thin YAML job launcher and monitoring commands."""

from pathlib import Path
from types import SimpleNamespace
import sys

import yaml

from tc.jobs import Job, load_job


def parse_args(argv: list[str] | None = None) -> Job | SimpleNamespace:
    """Load one YAML job or parse an explicit monitoring command."""
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if argv and argv[0] in ("status", "watch", "stop"):
            if len(argv) not in (2, 3):
                raise ValueError("usage: python -m tc {status,watch,stop} JOB_ID [JOBS_DIR]")
            return SimpleNamespace(command=argv[0], job_id=argv[1], jobs_dir=argv[2] if len(argv) == 3 else "runs")
        names = [arg[7:] for arg in argv if arg.startswith("--name=")]
        paths = [arg for arg in argv if not arg.startswith("--name=")]
        if len(paths) != 1 or len(names) > 1 or paths[0].startswith("-"):
            raise ValueError("usage: python -m tc [--name=NAME] JOB.yaml")
        config = yaml.safe_load(Path(paths[0]).read_text())
        if names:
            if not isinstance(config, dict):
                raise ValueError("job configuration must be a mapping")
            config["name"] = names[0]
        job = load_job(config)
        job.original_argv = argv
        return job
    except (OSError, ValueError, TypeError, yaml.YAMLError) as error:
        raise SystemExit(str(error)) from error
