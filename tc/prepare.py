"""Select question JSONL while preserving IDs and seed indices."""

import json
from pathlib import Path
from tc.data import select, union
from tc.job_utils import read_rows
from dataclasses import dataclass, field
from typing import ClassVar
from tc.job import ExecutionContext, Job

@dataclass
class Prepare(Job):
    """Filter and sample question JSONL, then combine selected rows in input order."""
    command: ClassVar[str] = "prepare"
    count: int | None = None
    seed: int = 42
    exclude: list[str] = field(default_factory=list)
    split: str = "train"

    def __post_init__(self) -> None:
        """Validate this job's configuration before execution."""
        super().__post_init__()
        self.require_positive("count")

    def run(self, context: ExecutionContext) -> None:
        directory, record, artifacts = context.directory, context.record, context.artifacts
        if not self.input:
            raise ValueError("prepare requires --input question JSONL")
        collections = []
        for source in self.input:
            rows = read_rows(source)
            for row in rows:
                if not isinstance(row.get("messages"), list) or not row["messages"]:
                    raise ValueError("questions require nonempty messages")
                if type(row.get("seed_index")) is not int or row["seed_index"] < 0:
                    raise ValueError("questions require a nonnegative integer seed_index")
            collections.append(select(rows, split=self.split, exclude=self.exclude,
                                      count=self.count, seed=self.seed))
        rows = union(collections)
        if not rows:
            raise ValueError("selection contains no questions")
        destination = directory / "outputs" / "questions.jsonl"
        with destination.open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        artifacts.extend([destination])
