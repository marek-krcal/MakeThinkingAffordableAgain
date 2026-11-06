"""Immutable job records and committed runtime snapshots."""

from __future__ import annotations
from typing import Any

from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess
from urllib.parse import urlsplit
from uuid import uuid4

import yaml

UPSTREAM_COMMIT = "bd52d8d54b6464478516dfa40458e73612926f2f"
COMMIT = re.compile(r"[0-9a-f]{40}")
# A conservative whole-checkout check avoids overlooking a newly added runtime file.


def git(repo: str | Path, *args: Any) -> str:
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


def resolved_revisions(args: Any, resolve: Any=resolve_revision) -> dict[str, Any]:
    """Resolve model, tokenizer, dataset and reference identities."""
    revisions = {}
    for key, kind in (("model", "model"), ("tokenizer", "model"), ("dataset", "dataset")):
        name = getattr(args, key, None)
        if name:
            requested = getattr(args, key + "_revision", None) or "main"
            revisions[key] = {"id": name, "revision": resolve(kind, name, requested)}
    if args.command == "train" and args.algorithm == "dpo":
        revisions["reference"] = dict(revisions["model"])
    return revisions


def make_record(args: Any, repo: str | Path, resolve: Any=resolve_revision, parent_id: str | None = None) -> dict[str, Any]:
    """Create a fresh job identity with settings and source provenance."""
    if args.command in ("status", "watch", "stop"):
        raise ValueError("monitoring commands do not create jobs")
    return {
        "job_id": str(uuid4()),
        "name": args.name or args.command,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parent_id": parent_id,
        **source_identity(repo),
        "original_argv": list(getattr(args, "original_argv", [])),
        "arguments": args.config(),
        "revisions": resolved_revisions(args, resolve),
    }


def write_record(record: dict[str, Any], directory: Path) -> None:
    """Create once; never replace another execution's configuration."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "config.yaml").open("x", encoding="utf-8") as stream:
        yaml.safe_dump(record, stream, sort_keys=False, default_flow_style=False)


def archive_source(record: dict[str, Any], repo: str | Path, destination: str | Path) -> None:
    """Workers must execute this commit archive, never the live checkout."""
    with Path(destination).open("xb") as stream:
        subprocess.run(
            ["git", "-C", str(repo), "archive", "--format=tar", record["source_commit"]],
            stdout=stream, check=True,
        )
