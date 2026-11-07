"""Bounded generation using upstream prompts, seeds and decoding."""

from __future__ import annotations
from typing import Any

import json
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import yaml

import evaluate as upstream
from tc.data import union
from tc.job_utils import check_stop, progress, read_rows, write_json


def completed_records(path: Path) -> list[dict[str, Any]]:
    """Read committed responses, ignoring an unfinished final append."""
    records = []
    with path.open("rb") as stream:
        for line in stream:
            if not line.endswith(b"\n"):
                break
            if line.strip():
                records.append(json.loads(line))
    return records


def adapter_fingerprint(directory: str | None) -> str | None:
    """Hash adapter configuration and weights independently of their directory."""
    if directory is None:
        return None
    root = Path(directory)
    weights = sorted([*root.glob("*.safetensors"), *root.glob("*.bin")])
    if not weights:
        raise ValueError("adapter weights are missing")
    digest = hashlib.sha256()
    for path in [root / "adapter_config.json", *weights]:
        with path.open("rb") as stream:
            digest.update(path.name.encode() + b"\0" + hashlib.file_digest(stream, "sha256").digest())
    return digest.hexdigest()


def pending_jobs(jobs: Any, records: list[dict[str, Any]]) -> Any:
    """Validate saved response identities and return missing samples."""
    expected = {(j["row"]["id"], j["sample"]): j for j in jobs}
    seen = set()
    for record in records:
        key = (record["id"], record["sample_index"])
        if key in seen or key not in expected or record["seed"] != expected[key]["seed"]:
            raise ValueError("resume records have duplicate or incompatible identities")
        if not isinstance(record.get("token_ids"), list):
            raise ValueError("resume records must preserve completion tokens")
        seen.add(key)
    return [j for j in jobs if (j["row"]["id"], j["sample"]) not in seen]


import math
from dataclasses import dataclass
from typing import ClassVar
from tc.job import ExecutionContext, ModelJob

@dataclass
class Generate(ModelJob):
    """Generate bounded batches with upstream decoding and resume saved responses."""
    command: ClassVar[str] = "generate"
    seed: int = 9001
    adapter: str | None = None
    resume: str | None = None
    max_tokens: int | None = None
    responses: int = 8
    temperature: float = .6
    top_p: float = .95
    top_k: int = 20
    max_num_seqs: int = 128
    max_num_batched_tokens: int = 8192
    request_batch_size: int = 128
    max_model_len: int = 12288
    gpu_memory_utilization: float = .9

    def __post_init__(self) -> None:
        """Validate sampling limits and resolve model defaults."""
        super().__post_init__()
        self.require_positive("responses", "max_tokens", "max_num_seqs", "max_num_batched_tokens",
                              "request_batch_size", "max_model_len")
        if not 0 < self.top_p <= 1 or not 0 < self.gpu_memory_utilization < 1:
            raise ValueError("invalid top_p or gpu_memory_utilization")
        if not math.isfinite(self.temperature) or self.temperature < 0 or self.top_k < -1:
            raise ValueError("invalid temperature or top_k")

    def run(self, context: ExecutionContext) -> None:
        directory, record, artifacts = context.directory, context.record, context.artifacts
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        questions = union([read_rows(path) for path in self.input])
        if not questions:
            raise ValueError("generate requires nonempty question inputs")
        revisions = record["revisions"]
        tokenizer = AutoTokenizer.from_pretrained(revisions["tokenizer"]["id"],
                                                 revision=revisions["tokenizer"]["revision"])
        options = SimpleNamespace(samples=self.responses, seed=self.seed, max_tokens=self.max_tokens)
        jobs = upstream.build_jobs(questions, tokenizer, options)
        for job in jobs:
            cap = upstream.token_cap(job["row"]["dataset"], options)
            if len(job["ids"]) > upstream.PROMPT_BUDGET or len(job["ids"]) + cap > self.max_model_len:
                raise ValueError("prompt/output exceeds context capacity; no truncation performed")
        prompts = [{"question": q, "token_ids": jobs[i * self.responses]["ids"]}
                   for i, q in enumerate(questions)]
        write_json(directory / "outputs/prompts.json", prompts)
        identity = {"adapter_sha256": adapter_fingerprint(self.adapter)}
        write_json(directory / "outputs/model-identity.json", identity)
        completed = []
        if self.resume:
            previous = Path(self.resume)
            old = yaml.safe_load((previous / "config.yaml").read_text())
            ignored = {"name", "jobs_dir", "input", "resume", "local"}
            current = {k: v for k, v in record["arguments"].items() if k not in ignored}
            prior = {k: v for k, v in old["arguments"].items() if k not in ignored}
            if current != prior or revisions != old["revisions"]:
                raise ValueError("resume generation settings differ")
            if json.loads((previous / "outputs/prompts.json").read_text()) != prompts:
                raise ValueError("resume prompts differ")
            saved_identity = previous / "outputs/model-identity.json"
            prior_identity = json.loads(saved_identity.read_text()) if saved_identity.exists() else {"adapter_sha256": None}
            if prior_identity != identity:
                raise ValueError("resume adapter identity is missing or differs")
            completed = completed_records(previous / "outputs/records.jsonl")
        jobs = pending_jobs(jobs, completed)
        output_tokens = sum(len(row["token_ids"]) for row in completed)
        destination = directory / "outputs/records.jsonl"
        with destination.open("x") as stream:
            for item in completed:
                stream.write(json.dumps(item) + "\n")
            stream.flush()
            progress(directory, completed=len(completed), total=len(completed) + len(jobs),
                     output_tokens=output_tokens)
            if not jobs:
                artifacts.extend([destination, directory / "outputs/prompts.json", directory / "outputs/model-identity.json"])
                return
            eos = tokenizer.eos_token_id
            eos_ids = set(eos if isinstance(eos, list) else [eos])
            open_id, close_id = (tokenizer.convert_tokens_to_ids(t) for t in ("<think>", "</think>"))
            rank = json.loads((Path(self.adapter) / "adapter_config.json").read_text())["r"] if self.adapter else 16
            os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
            engine = LLM(model=revisions["model"]["id"], revision=revisions["model"]["revision"],
                         tokenizer=revisions["tokenizer"]["id"], tokenizer_revision=revisions["tokenizer"]["revision"],
                         dtype="float16", tensor_parallel_size=1, max_model_len=self.max_model_len,
                         max_num_seqs=self.max_num_seqs, max_num_batched_tokens=self.max_num_batched_tokens,
                         gpu_memory_utilization=self.gpu_memory_utilization, enable_prefix_caching=False,
                         enforce_eager=False, generation_config="vllm", seed=self.seed,
                         enable_lora=bool(self.adapter), max_lora_rank=rank)
            lora = LoRARequest("candidate", 1, str(Path(self.adapter).resolve())) if self.adapter else None
            for start in range(0, len(jobs), self.request_batch_size):
                check_stop(directory)
                batch = jobs[start:start + self.request_batch_size]
                settings = [SamplingParams(max_tokens=upstream.token_cap(j["row"]["dataset"], options),
                                           seed=j["seed"], stop_token_ids=list(eos_ids),
                                           temperature=self.temperature, top_p=self.top_p, top_k=self.top_k)
                            for j in batch]
                outputs = engine.generate([{"prompt_token_ids": j["ids"]} for j in batch], settings,
                                          lora_request=lora)
                if len(outputs) != len(batch):
                    raise ValueError("generation output count differs")
                for job, output in zip(batch, outputs):
                    row = upstream.to_record(job, output, lora, tokenizer, eos_ids, open_id, close_id)
                    row.update(token_ids=list(output.outputs[0].token_ids),
                               finish_reason=output.outputs[0].finish_reason)
                    output_tokens += len(row["token_ids"])
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                progress(directory, completed=len(completed) + start + len(batch),
                         total=len(completed) + len(jobs), output_tokens=output_tokens)
        artifacts.extend([destination, directory / "outputs/prompts.json", directory / "outputs/model-identity.json"])
