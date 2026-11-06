"""Stateless source provenance, revision lookup and job file helpers."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
import json
import re
import subprocess
import yaml

UPSTREAM_COMMIT = "bd52d8d54b6464478516dfa40458e73612926f2f"
COMMIT = re.compile(r"[0-9a-f]{40}")
# A conservative whole-checkout check avoids overlooking a newly added runtime file.


def git(repo: str | Path, *args: str) -> str:
    """Run a Git query in the selected repository."""
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def repository_url(remote: str) -> str:
    """Normalize a GitHub remote without credentials or query parameters."""
    if remote.startswith("git@github.com:"):
        remote = "https://github.com/" + remote.split(":", 1)[1]
    url = urlsplit(remote)
    if url.scheme != "https" or url.hostname != "github.com":
        raise ValueError("origin must identify an HTTPS or SSH GitHub repository")
    path = url.path.removesuffix(".git").strip("/")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", path):
        raise ValueError("invalid GitHub repository path")
    return "https://github.com/" + path


def source_identity(repo: str | Path) -> dict[str, Any]:
    """Record the executing commit and require a clean checkout."""
    commit = git(repo, "rev-parse", "HEAD")
    git(repo, "merge-base", "--is-ancestor", UPSTREAM_COMMIT, commit)
    dirty = bool(git(repo, "status", "--porcelain", "--untracked-files=all"))
    if dirty:
        raise ValueError("jobs require committed source; commit or remove working changes")
    return {
        "repository_url": repository_url(git(repo, "remote", "get-url", "origin")),
        "source_commit": commit,
        "upstream_commit": UPSTREAM_COMMIT,
        "source_dirty": dirty,
    }


def resolve_revision(kind: str, name: str, revision: str) -> str:
    # Import lazily: CPU-side configuration needs no training dependencies.
    """Resolve a Hub reference to an immutable commit."""
    from huggingface_hub import HfApi

    info = HfApi().repo_info(name, repo_type=kind, revision=revision)
    if not info.sha or not COMMIT.fullmatch(info.sha):
        raise ValueError(f"could not resolve immutable revision for {name}")
    return info.sha


def check_stop(directory: Path) -> None:
    """Raise when this job has received a cooperative stop request."""
    if (Path(directory) / "stop.request").exists():
        raise KeyboardInterrupt("stop requested")


def progress(directory: Path, **fields: Any) -> None:
    """Atomically merge progress fields into the current job status."""
    path = Path(directory) / "status.json"
    status = json.loads(path.read_text()) if path.exists() else {}
    status.update(fields, updated_at=datetime.now(timezone.utc).isoformat())
    write_json(path, status)


def read_rows(path: Path) -> Any:
    """Read nonempty JSONL records, rejecting malformed input."""
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path: Path, value: Any) -> None:
    """Replace a JSON file atomically."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def write_record(record: dict[str, Any], directory: Path) -> None:
    """Create once; never replace another execution's configuration."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "config.yaml").open("x", encoding="utf-8") as stream:
        yaml.safe_dump(record, stream, sort_keys=False, default_flow_style=False)


def archive_source(record: dict[str, Any], repo: str | Path, destination: str | Path) -> None:
    """Archive the recorded source commit rather than working-tree changes."""
    with Path(destination).open("xb") as stream:
        subprocess.run(
            ["git", "-C", str(repo), "archive", "--format=tar", record["source_commit"]],
            stdout=stream, check=True,
        )
