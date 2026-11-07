"""Saved-answer adapters around unchanged upstream grading and statistics."""

from __future__ import annotations
from typing import Any

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import yaml

import evaluate as upstream
from tc.data import strict_pairs
from tc.job_utils import read_rows, write_json
from utils.eval.compare import compare as upstream_compare


def indexed(records: list[dict[str, Any]]) -> Any:
    """Index responses by question and sample, rejecting duplicates."""
    result = {}
    for row in records:
        key = (row["id"], row["sample_index"])
        if key in result:
            raise ValueError("duplicate response identity")
        result[key] = row
    return result


from dataclasses import dataclass
from typing import ClassVar
from tc.job import ExecutionContext, Job

@dataclass
class Evaluate(Job):
    """Grade a complete saved response cohort with unchanged upstream evaluators."""
    command: ClassVar[str] = "evaluate"
    questions: str | None = None
    workers: int = 1
    sandbox: str | None = None

    def run(self, context: ExecutionContext) -> None:
        directory, record, artifacts = context.directory, context.record, context.artifacts
        if len(self.input) != 1:
            raise ValueError("evaluate requires one generation job directory")
        source = Path(self.input[0])
        config = yaml.safe_load((source / "config.yaml").read_text())
        generated = config["arguments"]
        prompts = json.loads((source / "outputs/prompts.json").read_text())
        questions = [p["question"] for p in prompts]
        records = read_rows(source / "outputs/records.jsonl")
        expected = {(q["id"], sample) for q in questions for sample in range(generated["responses"])}
        if set(indexed(records)) != expected:
            raise ValueError("generation cohort is incomplete")
        for row in records:
            for key in ("correct", "parse_error", "grading_error"):
                row.pop(key, None)
        wanted = {q["grader"] for q in questions}
        suite = upstream.load_suite(Path(self.questions)) if self.questions else {}
        if wanted - {"numeric"} and not suite:
            raise ValueError("nonnumeric grading requires --questions upstream benchmark assets")
        options = SimpleNamespace(benchmarks=sorted(wanted), workers=self.workers, sandbox=self.sandbox)
        if "ifeval" in wanted:
            import nltk
            for package in ("punkt_tab", "punkt"):
                if not nltk.download(package, quiet=True):
                    raise RuntimeError(f"Missing IFEval resource: {package}")
        full = bool(suite) and questions == read_rows(Path(self.questions) / "questions.jsonl")
        cohort_hash = suite["cohort_sha256"] if full else hashlib.sha256(json.dumps(questions, sort_keys=True).encode()).hexdigest()
        benchmark = {dataset: name for name, datasets in upstream.BENCHMARKS.items() for dataset in datasets}
        grading = upstream.grading_context(suite, options)
        upstream.grade_all(questions, records, grading)
        output = directory / "outputs"
        upstream.save_records(output, records)
        # Persist ungraded answers before allowing aggregation to reject a failed grader.
        results, failed = upstream.collect(questions, records, generated["responses"])
        sampling = {k: generated[k] for k in ("temperature", "top_p", "top_k")}
        caps = {q["dataset"]: upstream.token_cap(q["dataset"], SimpleNamespace(max_tokens=generated["max_tokens"]))
                for q in questions}
        metadata = {"model": generated["model"], "revision": config["revisions"]["model"]["revision"],
                    "adapter": generated["adapter"], "benchmarks": sorted({benchmark.get(q["dataset"], q["dataset"]) for q in questions}),
                    "samples": generated["responses"], "seed": generated["seed"], "sampling": sampling,
                    "max_tokens": caps, "max_model_len": generated["max_model_len"],
                    "cohort_sha256": cohort_hash,
                    "cohort_total": len(questions), "selection": "full" if full else "saved", "smoke": False,
                    "revisions": suite.get("revisions", {}), "results": results, "failed_benchmarks": failed,
                    "sandbox": {"mode": grading["mode"], "why": grading["why"]}}
        write_json(output / "results.json", metadata)
        if failed:
            raise ValueError("grading incomplete; saved answers can be regraded")
        artifacts.extend([output / "records.jsonl", output / "results.json"])

    def __post_init__(self) -> None:
        """Validate this job's configuration before execution."""
        super().__post_init__()
        self.require_positive("workers")
        if self.sandbox not in (None, "bwrap", "none"):
            raise ValueError("invalid sandbox")


@dataclass
class Compare(Job):
    """Validate matching response cohorts and compare upstream evaluation results."""
    command: ClassVar[str] = "compare"

    def run(self, context: ExecutionContext) -> None:
        directory, record, artifacts = context.directory, context.record, context.artifacts
        if len(self.input) != 2:
            raise ValueError("compare requires baseline and candidate evaluation directories")
        paths = [Path(p) / "outputs" for p in self.input]
        left, right = [indexed(read_rows(p / "records.jsonl")) for p in paths]
        if set(left) != set(right):
            raise ValueError("comparison requires identical complete response cohorts")
        for key in left:
            if any(left[key][field] != right[key][field] for field in ("dataset", "seed")):
                raise ValueError("comparison response identities differ")
            if left[key].get("correct") is None or right[key].get("correct") is None:
                raise ValueError("comparison contains ungraded responses")
        metadata = [json.loads((p / "results.json").read_text()) for p in paths]
        for field in ("sampling", "max_model_len"):
            if metadata[0][field] != metadata[1][field]:
                raise ValueError(f"comparison {field} differs")
        destination = directory / "outputs/comparison.json"
        write_json(destination, upstream_compare(*paths))
        artifacts.extend([destination])
