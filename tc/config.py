"""Serializable jobs; defaults live beside their inputs."""

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar, get_args, get_origin, get_type_hints
from types import UnionType
import math

import yaml

MODEL = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


@dataclass
class Job:
    """Serializable settings shared by concrete jobs."""
    command: ClassVar[str]
    name: str = ""
    local: bool = False
    jobs_dir: str = "runs"
    input: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate declared field types for YAML and direct Python construction."""
        def matches(value: Any, annotation: Any) -> bool:
            """Check a YAML value against its declared field type."""
            origin = get_origin(annotation)
            if origin is UnionType:
                return any(matches(value, item) for item in get_args(annotation))
            if origin is list:
                return isinstance(value, list) and all(matches(item, get_args(annotation)[0]) for item in value)
            if annotation is float:
                return type(value) in (int, float) and math.isfinite(value)
            return type(value) is annotation

        annotations = get_type_hints(type(self))
        for member in fields(self):
            if not matches(getattr(self, member.name), annotations[member.name]):
                raise ValueError(f"invalid field: {member.name}")
        self.name = self.name or self.command

    def require_positive(self, *names: str) -> None:
        """Reject nonpositive values for the supplied optional numeric fields."""
        for name in names:
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")

    def config(self) -> dict[str, Any]:
        """Return serializable settings including the job type."""
        return {"command": self.command, **asdict(self)}

    def save(self, path: str | Path) -> None:
        """Write resolved settings as YAML."""
        Path(path).write_text(yaml.safe_dump(self.config(), sort_keys=False))


@dataclass
class ModelJob(Job):
    """Job settings with model and tokenizer revisions resolved by default."""
    model: str = MODEL
    model_revision: str | None = None
    tokenizer: str | None = None
    tokenizer_revision: str | None = None

    def __post_init__(self) -> None:
        """Fill missing model and tokenizer revision defaults."""
        super().__post_init__()
        self.model_revision = self.model_revision or (MODEL_REVISION if self.model == MODEL else "main")
        self.tokenizer = self.tokenizer or self.model
        self.tokenizer_revision = self.tokenizer_revision or (self.model_revision if self.tokenizer == self.model else "main")


def load_job(config: dict[str, Any]) -> Job:
    """Construct the selected dataclass from a YAML mapping."""
    from tc.jobs import JOBS

    if not isinstance(config, dict):
        raise ValueError("job configuration must be a mapping")
    values = dict(config)
    kind = values.pop("command", None)
    if not isinstance(kind, str) or kind not in JOBS:
        raise ValueError(f"unknown job: {kind}")
    try:
        return JOBS[kind](**values)
    except TypeError as error:
        raise ValueError(str(error)) from error
