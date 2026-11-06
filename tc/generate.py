"""Serializable job settings."""

from dataclasses import dataclass, field
from typing import ClassVar
import math
from tc.config import Job, ModelJob


@dataclass
class Generate(ModelJob):
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
        super().__post_init__()
        self.require_positive("responses", "max_tokens", "max_num_seqs", "max_num_batched_tokens",
                              "request_batch_size", "max_model_len")
        if not 0 < self.top_p <= 1 or not 0 < self.gpu_memory_utilization < 1:
            raise ValueError("invalid top_p or gpu_memory_utilization")
        if not math.isfinite(self.temperature) or self.temperature < 0 or self.top_k < -1:
            raise ValueError("invalid temperature or top_k")
