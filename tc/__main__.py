"""Record a configured job before worker execution is added."""
from pathlib import Path
from tc.cli import parse_args
from tc.job import make_record, write_record

def main() -> None:
    job = parse_args()
    record = make_record(job, Path(__file__).resolve().parent.parent)
    directory = Path(job.jobs_dir).resolve() / record["job_id"]
    write_record(record, directory)
    print(directory)

if __name__ == "__main__":
    main()
