"""Serializable job settings."""

from dataclasses import dataclass, field
from typing import ClassVar
import math
from tc.config import Job, ModelJob


@dataclass
class Prepare(Job):
    command: ClassVar[str] = "prepare"
    generation_job: str | None = None
    count: int | None = None
    seed: int = 42
    exclude: list[str] = field(default_factory=list)
    dataset: str | None = None
    dataset_revision: str | None = None
    split: str = "train"

    def __post_init__(self) -> None:
        """Validate this job's configuration before execution."""
        super().__post_init__()
        self.require_positive("count")
        if self.dataset_revision and not self.dataset:
            raise ValueError("dataset_revision requires dataset")

