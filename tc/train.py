"""One-device PEFT training with restartable optimizer updates."""

from __future__ import annotations
from typing import Any

import hashlib
import json
from pathlib import Path
import random
import shutil


from tc.sampling import batches, learning_rate
from tc.job_utils import check_stop, progress


def save_checkpoint(path: Path, model: Any, optimizer: Any, scaler: Any, step: int, identity: dict[str, Any]) -> None:
    """Atomically save adapter, optimizer, scaler and RNG state."""
    import torch

    temporary = path.with_name(path.name + ".partial")
    temporary.mkdir(parents=True)
    model.save_pretrained(temporary)
    torch.save({"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                "step": step, "identity": identity, "random": random.getstate(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
               temporary / "state.pt")
    temporary.rename(path)


def restore_checkpoint(path: Path, model: Any, optimizer: Any, scaler: Any, identity: dict[str, Any]) -> int:
    """Validate checkpoint identity and restore exact training state."""
    import torch
    from peft import load_peft_weights, set_peft_model_state_dict

    state = torch.load(Path(path) / "state.pt", map_location="cpu", weights_only=True)
    if state["identity"] != identity:
        raise ValueError("checkpoint training settings or input differ")
    set_peft_model_state_dict(model, load_peft_weights(path, device="cpu"))
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    random.setstate(state["random"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    return state["step"]


import math
from dataclasses import dataclass, field
from typing import ClassVar
from tc.job import ExecutionContext, ModelJob
from tc.preferences import PreferenceGroups

@dataclass
class Train(ModelJob):
    """Train single-device LoRA with SimPO or DPO and restartable optimizer updates."""
    command: ClassVar[str] = "train"
    input: ClassVar[None] = None
    groups: PreferenceGroups = field(default_factory=PreferenceGroups)
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
        """Validate objective and optimizer settings and fill recipe defaults."""
        if isinstance(self.groups, dict):
            self.groups = PreferenceGroups(**self.groups)
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

    def run(self, context: ExecutionContext) -> None:
        directory, record, artifacts = context.directory, context.record, context.artifacts
        import torch
        from tc.objectives import group_loss
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM

        groups = self.groups.form_groups()
        destination = directory / "outputs/groups.jsonl"
        destination.write_text("".join(json.dumps(group) + "\n" for group in groups))
        artifacts.append(destination)
        # Validate the sampler before loading model weights.
        batches(groups, seed=self.seed, batch_size=self.batch_size, steps=self.steps, epochs=self.epochs)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        revision = record["revisions"]["model"]
        model = AutoModelForCausalLM.from_pretrained(revision["id"], revision=revision["revision"],
                                                   torch_dtype=torch.float16 if self.device == "cuda" else torch.float32)
        model.to(self.device)
        model = get_peft_model(model, LoraConfig(r=self.lora_rank, lora_alpha=self.lora_alpha,
                                               lora_dropout=self.lora_dropout, task_type="CAUSAL_LM",
                                               target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        excluded = {"input", "resume", "name", "jobs_dir", "local", "checkpoint_every", "retain_step"}
        arguments = {k: v for k, v in record["arguments"].items() if k not in excluded}
        arguments["groups"] = {k: v for k, v in arguments["groups"].items() if k != "input"}
        identity = {"arguments": arguments,
                    "revisions": record["revisions"],
                    "input_sha256": hashlib.sha256(json.dumps(groups, sort_keys=True).encode()).hexdigest()}
        with (directory / "log.txt").open("x") as log:
            schedule = batches(groups, seed=self.seed, batch_size=self.batch_size,
                               steps=self.steps, epochs=self.epochs)
            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                         lr=self.learning_rate, weight_decay=self.weight_decay)
            cuda = next(model.parameters()).device.type == "cuda"
            scaler = torch.amp.GradScaler("cuda", enabled=cuda)
            start = restore_checkpoint(self.resume, model, optimizer, scaler, identity) if self.resume else 0
            if start > len(schedule):
                raise ValueError("checkpoint exceeds requested schedule")
            checkpoint_dir = Path(directory) / "checkpoints"
            checkpoint_dir.mkdir()
            retained = set(self.retain_step if self.retain_step is not None else
                           ([737, 860, 982] if len(schedule) == 982 else [len(schedule)]))
            retained.add(len(schedule))
            model.train()
            for offset in range(start, len(schedule)):
                check_stop(directory)
                step, batch = offset + 1, schedule[offset]
                optimizer.zero_grad(set_to_none=True)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = learning_rate(step, len(schedule), self.learning_rate)
                micro_size = max(1, (len(batch) + self.gradient_accumulation - 1) // self.gradient_accumulation)
                loss_value = 0.
                for begin in range(0, len(batch), micro_size):
                    with torch.autocast(device_type="cuda" if cuda else "cpu", dtype=torch.float16, enabled=cuda):
                        losses = [group_loss(model, group, algorithm=self.algorithm, beta=self.beta, gamma=self.gamma)
                                  for group in batch[begin:begin + micro_size]]
                        loss = torch.stack(losses).sum() / len(batch)
                    loss_value += float(loss.detach())
                    scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                if step in retained or step % self.checkpoint_every == 0 or step == len(schedule):
                    save_checkpoint(checkpoint_dir / f"step-{step}", model, optimizer, scaler, step, identity)
                    progress(directory, checkpoint=f"checkpoints/step-{step}")
                    safety = sorted((p for p in checkpoint_dir.glob("step-*") if int(p.name[5:]) not in retained),
                                    key=lambda p: int(p.name[5:]))
                    for old in safety[:-1]:
                        shutil.rmtree(old)
                log.write(json.dumps({"step": step, "loss": loss_value}) + "\n")
                log.flush()
                progress(directory, step=step, total_steps=len(schedule), loss=loss_value)
            model.save_pretrained(Path(directory) / "outputs/adapter")
            for checkpoint in checkpoint_dir.glob("step-*"):
                if int(checkpoint.name[5:]) not in retained:
                    shutil.rmtree(checkpoint)
            artifacts.extend([Path(directory) / "outputs/adapter", checkpoint_dir])
