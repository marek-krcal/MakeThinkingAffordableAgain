"""Serializable job settings."""

from dataclasses import dataclass, field
from typing import ClassVar
import math
from tc.job import Job, ModelJob


@dataclass
class Train(ModelJob):
    command: ClassVar[str] = "train"
    seed: int = 42
    resume: str | None = None
    device: str = "cuda"
    checkpoint_every: int = 49
    retain_step: list[int] | None = None
    algorithm: str = "simpo"
    beta: float | None = None
    gamma: float | None = None
    steps: int | None = None
    epochs: int | None = None
    learning_rate: float = 1e-6
    weight_decay: float = .01
    batch_size: int = 8
    gradient_accumulation: int = 8
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.

    def __post_init__(self) -> None:
        super().__post_init__()
        self.require_positive("steps", "epochs", "checkpoint_every", "batch_size",
                              "gradient_accumulation", "lora_rank", "lora_alpha")
        if self.retain_step is not None and any(step <= 0 for step in self.retain_step):
            raise ValueError("retain_step must contain positive steps")
        if self.algorithm not in ("simpo", "dpo") or self.device not in ("cpu", "cuda"):
            raise ValueError("invalid algorithm or device")
        if self.steps is not None and self.epochs is not None:
            raise ValueError("choose steps or epochs")
        if self.steps is None and self.epochs is None:
            self.steps = 982
        if self.algorithm == "dpo" and self.gamma is not None:
            raise ValueError("gamma is only supported by SimPO")
        self.beta = self.beta if self.beta is not None else (2. if self.algorithm == "simpo" else .1)
        if self.algorithm == "simpo" and self.gamma is None:
            self.gamma = 1.
        if not math.isfinite(self.beta) or self.beta <= 0:
            raise ValueError("beta must be finite and positive")
        if self.gamma is not None and (not math.isfinite(self.gamma) or self.gamma < 0):
            raise ValueError("gamma must be finite and nonnegative")
        if self.learning_rate <= 0 or self.weight_decay < 0 or not 0 <= self.lora_dropout < 1:
            raise ValueError("invalid optimizer or LoRA settings")
