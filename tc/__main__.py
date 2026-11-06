"""Run a configured job."""

from tc.cli import parse_args
from tc.jobs import Job


def main() -> None:
    """Launch a configured job or its monitoring command."""
    job = parse_args()
    if isinstance(job, Job):
        job.execute()
    else:
        from tc.monitor import monitor
        monitor(job)


if __name__ == "__main__":
    main()
