"""Build a training-only suite from audited, content-addressed question artifacts."""

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import shutil
from typing import Any, ClassVar

from tc.data import union
from tc.job import ExecutionContext, Job
from tc.job_utils import read_rows, write_json


def instruction_stratum(row: dict[str, Any]) -> tuple[int, int]:
    """Describe instruction complexity without using model answers."""
    words = len(row["original_prompt"].split())
    return len(row["instruction_id_list"]), next((n for n in (30, 60, 100, 200, 300) if words <= n), 9999)


def sample_instructions(rows: list[dict[str, Any]], protected: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    """Match aggregate length/constraint strata, using the nearest available length bin."""
    target = Counter(instruction_stratum(row) for row in protected if row["grader"] == "ifeval")
    if not target:
        raise ValueError("instruction sampling requires protected cohort metadata")
    quotas = {key: int(n * count / target.total()) for key, n in target.items()}
    order = sorted(target, key=lambda key: (-(target[key] * count / target.total() - quotas[key]), key))
    for key in order[:count - sum(quotas.values())]:
        quotas[key] += 1
    pools = {}
    for row in rows:
        pools.setdefault(instruction_stratum(row), []).append(row)
    selected = []
    for index, (key, quota) in enumerate(sorted(quotas.items())):
        if not quota:
            continue
        choices = [other for other in pools if other[0] == key[0] and len(pools[other]) >= quota]
        if not choices:
            raise ValueError(f"insufficient audited instruction candidates for {key}")
        chosen = min(choices, key=lambda other: (abs(other[1] - key[1]), other))
        pool = sorted(pools[chosen], key=lambda row: row["id"])
        picked = random.Random(seed + index).sample(pool, quota)
        selected.extend(picked)
        identities = {row["id"] for row in picked}
        pools[chosen] = [row for row in pool if row["id"] not in identities]
    return selected


@dataclass
class PrepareExternal(Job):
    """Verify audited source artifacts, sample domains and build a separate training grader suite."""
    command: ClassVar[str] = "prepare-external"
    input: ClassVar[None] = None
    sources: str = ""
    protected: str = ""
    count: int = 200
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate source paths and the per-domain sample budget."""
        super().__post_init__()
        self.require_positive("count")

    def run(self, context: ExecutionContext) -> None:
        if not self.sources or not self.protected:
            raise ValueError("external preparation requires source manifest and protected suite")
        manifest = Path(self.sources)
        sources = json.loads(manifest.read_text())
        protected_root = Path(self.protected).resolve()
        protected_file = protected_root / "questions.jsonl"
        protected_digest = hashlib.sha256(protected_file.read_bytes()).hexdigest()
        original = json.loads((protected_root / "manifest.json").read_text())
        if protected_digest != sources["protected_sha256"] or protected_digest != original["cohort_sha256"]:
            raise ValueError("protected evaluation cohort changed")
        protected = read_rows(protected_file)
        protected_ids = {row["id"] for row in protected}
        questions, provenance = [], []
        problems = None
        domains = set()
        for source in sources["sources"]:
            if source["domain"] in domains:
                raise ValueError("one audited artifact per domain is required")
            domains.add(source["domain"])
            path = (manifest.parent / source["questions"]).resolve()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != source["questions_sha256"]:
                raise ValueError(f"audited question artifact changed: {path.name}")
            if not all(source.get(key) for key in ("revision", "raw_sha256", "license", "audit")):
                raise ValueError("source revision, raw digest, license and audit are required")
            audit = manifest.parent / source["audit"]
            if hashlib.sha256(audit.read_bytes()).hexdigest() != source["audit_sha256"]:
                raise ValueError("source audit changed")
            evidence = json.loads(audit.read_text())
            if any(evidence.get(key) != value for key, value in
                   (("source_sha256", source["raw_sha256"]), ("questions_sha256", digest),
                    ("protected_sha256", protected_digest))):
                raise ValueError("audit does not bind this source, artifact and protected cohort")
            rows = union([read_rows(path)])
            if source["domain"] not in ("mbpp", "ifeval", "bfcl") or any(row["grader"] != source["domain"] for row in rows):
                raise ValueError("source domain differs from its audited questions")
            if {row["id"] for row in rows} & protected_ids:
                raise ValueError("audited input contains protected evaluation identities")
            if any(row.get("split") != "train" or not row.get("messages") or
                   type(row.get("seed_index")) is not int or row["seed_index"] < 0 for row in rows):
                raise ValueError("audited questions require train split, messages and stable seeds")
            if source["domain"] == "ifeval":
                chosen = sample_instructions(rows, protected, self.count, self.seed)
            else:
                ordered = sorted(rows, key=lambda row: row["id"])
                chosen = random.Random(self.seed).sample(ordered, min(self.count, len(ordered)))
            questions.extend(chosen)
            provenance.append(dict(source, selected_ids=sorted(row["id"] for row in chosen)))
            if source["domain"] == "mbpp":
                problem_file = manifest.parent / source["problems"]
                if hashlib.sha256(problem_file.read_bytes()).hexdigest() != source["problems_sha256"]:
                    raise ValueError("validated code fixtures changed")
                problems = {row["task_id"]: row for row in read_rows(problem_file)}
                if not {row["id"] for row in chosen} <= problems.keys():
                    raise ValueError("selected code question lacks a validated fixture")
        questions = union([questions])
        if not questions:
            raise ValueError("empty external cohort")
        # Interleave domains for representative early throughput; preserve source seeds.
        questions.sort(key=lambda row: row["id"])
        random.Random(self.seed).shuffle(questions)
        suite = context.directory / "outputs/suite"
        suite.mkdir()
        shutil.copytree(protected_root / "vendor", suite / "vendor", ignore=shutil.ignore_patterns("__pycache__"))
        for path in ("bfcl/bfcl_eval/data", "instruction_following_eval/data"):
            shutil.rmtree(suite / "vendor" / path, ignore_errors=True)
        (suite / "raw").mkdir()
        if problems is not None:
            selected = [problems[row["id"]] for row in questions if row["grader"] == "mbpp"]
            (suite / "raw/mbpp.jsonl").write_text("".join(json.dumps(row) + "\n" for row in selected))
        destination = suite / "questions.jsonl"
        destination.write_text("".join(json.dumps(row) + "\n" for row in questions))
        revisions = {key: value for key, value in original["revisions"].items() if key != "mbpp"}
        revisions["mbpp"] = next((source["revision"] for source in provenance if source["domain"] == "mbpp"), None)
        write_json(suite / "manifest.json", {"cohort_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "total": len(questions),
                                            "revisions": revisions, "training_sources": provenance,
                                            "training_mbpp_cases": 3 if problems is not None else 0})
        write_json(context.directory / "outputs/sources.json", provenance)
        context.artifacts.extend([suite, context.directory / "outputs/sources.json"])
