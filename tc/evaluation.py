"""Serializable job settings."""

from dataclasses import dataclass, field
from typing import ClassVar
import math
from tc.config import Job, ModelJob


@dataclass
class Evaluate(Job):
    command: ClassVar[str] = "evaluate"
    questions: str | None = None
    workers: int = 1
    sandbox: str | None = None


    def __post_init__(self) -> None:
        """Validate this job's configuration before execution."""
        super().__post_init__()
        self.require_positive("workers")
        if self.sandbox not in (None, "bwrap", "none"):
            raise ValueError("invalid sandbox")


@dataclass
class Compare(Job):
    command: ClassVar[str] = "compare"
