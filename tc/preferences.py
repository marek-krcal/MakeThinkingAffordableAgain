"""Preference formation from graded answers and their original tokens."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json
import yaml

from tc.data import select, strict_pairs
from tc.job_utils import read_rows


@dataclass
class PreferenceGroups:
    """Verify graded generations and select groups containing strict preferences."""
    input: str = ""
    split: str = "train"
    count: int | None = None
    seed: int = 42
    exclude: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Reject invalid selection settings before training starts."""
        if type(self.input) is not str or type(self.split) is not str or type(self.seed) is not int:
            raise ValueError("invalid preference input, split or seed")
        if self.count is not None and (type(self.count) is not int or self.count <= 0):
            raise ValueError("preference count must be positive")
        if type(self.exclude) is not list or any(type(item) is not str for item in self.exclude):
            raise ValueError("preference exclusions must be question IDs")

    def form_groups(self) -> list[dict[str, Any]]:
        """Join complete grades to original tokens and select usable groups."""
        if not self.input:
            raise ValueError("groups.input requires an evaluation job directory")
        evaluation = Path(self.input)
        if json.loads((evaluation / "status.json").read_text())["state"] != "completed":
            raise ValueError("evaluation is not completed")
        results = json.loads((evaluation / "outputs/results.json").read_text())
        if results.get("failed_benchmarks"):
            raise ValueError("evaluation contains failed benchmarks")
        sources = yaml.safe_load((evaluation / "config.yaml").read_text())["arguments"]["input"]
        if len(sources) != 1:
            raise ValueError("evaluation requires one original generation")
        generation = Path(sources[0])
        prompts = json.loads((generation / "outputs/prompts.json").read_text())
        samples = yaml.safe_load((generation / "config.yaml").read_text())["arguments"]["responses"]
        expected = {(p["question"]["id"], sample) for p in prompts for sample in range(samples)}
        if len(expected) != len(prompts) * samples:
            raise ValueError("duplicate prompt identity")
        original = self.index(read_rows(generation / "outputs/records.jsonl"))
        graded = self.index(read_rows(evaluation / "outputs/records.jsonl"))
        if set(original) != expected or set(graded) != expected:
            raise ValueError("generation or grading cohort is incomplete")
        for key, row in graded.items():
            if any(row[field] != original[key][field] for field in ("seed", "dataset", "token_ids")):
                raise ValueError("graded response differs from saved generation")
            if type(row.get("correct")) is not bool or type(row["token_ids"]) is not list:
                raise ValueError("responses require grades and original completion tokens")
        groups = []
        for prompt in prompts:
            question = prompt["question"]
            responses = [{"sample_index": sample, "correct": graded[question["id"], sample]["correct"],
                          "completion_tokens": original[question["id"], sample]["token_ids"]}
                         for sample in range(samples)]
            if strict_pairs(responses):
                groups.append({"id": question["id"], "split": question["split"],
                               "prompt_tokens": prompt["token_ids"], "responses": responses})
        groups = select(groups, split=self.split, exclude=self.exclude, count=self.count, seed=self.seed)
        if not groups:
            raise ValueError("no training signal in selected responses")
        return groups

    @staticmethod
    def index(rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
        """Index each saved sample once, rejecting duplicate response identities."""
        result = {}
        for row in rows:
            key = (row["id"], row["sample_index"])
            if key in result:
                raise ValueError("duplicate response identity")
            result[key] = row
        return result
