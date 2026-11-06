"""Serializable jobs, source provenance and recorded execution lifecycle."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar, get_args, get_origin, get_type_hints
from types import UnionType
from datetime import datetime, timezone
from contextlib import contextmanager
from collections.abc import Iterator
from importlib.metadata import PackageNotFoundError, version
from uuid import uuid4
import json
import math
import platform
import traceback
import yaml

from tc.job_utils import resolve_revision, source_identity, write_json, write_record


@dataclass
class ExecutionContext:
    """Files, provenance and registered artifacts belonging to one execution."""
    directory: Path
    record: dict[str, Any]
    artifacts: list[Path] = field(default_factory=list)


MODEL = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


@dataclass
class Job:
    """Configure a job and execute its run method with recorded lifecycle state."""
    command: ClassVar[str]
    name: str = ""
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

    def run(self, context: ExecutionContext) -> None:
        raise NotImplementedError

    def execute(self) -> Path:
        """Run this job inside its recorded execution lifecycle."""
        with self.execution_context() as context:
            self.run(context)
        return context.directory

    @contextmanager
    def execution_context(self) -> Iterator[ExecutionContext]:
        """Create immutable records and persist terminal status around a job body."""
        record = self.make_record(Path(__file__).resolve().parent.parent)
        directory = Path(self.jobs_dir).resolve() / record["job_id"]
        write_record(record, directory)
        print(directory, flush=True)
        artifacts = []
        directory = Path(directory)
        (directory / "outputs").mkdir()
        write_json(directory / "environment.json", {"python": platform.python_version(),
                                                   "platform": platform.platform()})
        packages = {}
        for name in ("torch", "transformers", "peft", "vllm", "PyYAML", "huggingface-hub"):
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                pass
        environment = json.loads((directory / "environment.json").read_text())
        environment["packages"] = packages
        if "torch" in packages:
            import torch
            environment["gpu"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        write_json(directory / "environment.json", environment)
        status = {"job_id": record["job_id"], "state": "running"}
        def save() -> None:
            """Persist the current lifecycle state with its update time."""
            status["updated_at"] = datetime.now(timezone.utc).isoformat()
            write_json(directory / "status.json", status)
        save()
        try:
            yield ExecutionContext(directory, record, artifacts)
            write_json(directory / "artifacts.json", [str(p.relative_to(directory)) for p in artifacts])
            status["state"] = "completed"
        except BaseException as error:
            status.update(state="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                          error=str(error))
            raise
        finally:
            latest = json.loads((directory / "status.json").read_text())
            status = {**latest, **status}
            save()

    def resolved_revisions(self, resolve: Any=resolve_revision) -> dict[str, Any]:
        """Return no Hub revisions for jobs without a model."""
        return {}

    def make_record(self, repo: str | Path, resolve: Any=resolve_revision, parent_id: str | None = None) -> dict[str, Any]:
        """Create a fresh job identity with settings and source provenance."""
        if self.command in ("status", "watch", "stop"):
            raise ValueError("monitoring commands do not create jobs")
        return {
            "job_id": str(uuid4()),
            "name": self.name or self.command,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "parent_id": parent_id,
            **source_identity(repo),
            "original_argv": list(getattr(self, "original_argv", [])),
            "arguments": self.config(),
            "revisions": self.resolved_revisions(resolve),
        }


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

    def resolved_revisions(self, resolve: Any=resolve_revision) -> dict[str, Any]:
        """Resolve model, tokenizer and optional DPO reference identities."""
        revisions = {}
        for key, kind in (("model", "model"), ("tokenizer", "model")):
            name = getattr(self, key, None)
            if name:
                requested = getattr(self, key + "_revision", None) or "main"
                revisions[key] = {"id": name, "revision": resolve(kind, name, requested)}
        if self.command == "train" and self.algorithm == "dpo":
            revisions["reference"] = dict(revisions["model"])
        return revisions


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


