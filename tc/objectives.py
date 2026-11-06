"""Completion-only scores and pair-mean preference objectives."""

from __future__ import annotations
from typing import Any

import math

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from tc.data import strict_pairs


def preference_loss(sums: Any, lengths: Any, pairs: Any, *, algorithm: str, beta: float, gamma: float | None = None, reference: Any=None) -> Any:
    """Compute pair-mean SimPO or DPO loss from completion scores."""
    if not pairs or not math.isfinite(beta) or beta <= 0:
        raise ValueError("strict pairs and finite positive beta required")
    if sums.ndim != 1 or lengths.shape != sums.shape or (lengths <= 0).any():
        raise ValueError("one score and positive completion length per response required")
    if not torch.isfinite(sums).all():
        raise ValueError("nonfinite completion score")
    indices = torch.tensor(pairs, device=sums.device, dtype=torch.long)
    if (indices < 0).any() or (indices >= len(sums)).any() or (indices[:, 0] == indices[:, 1]).any():
        raise ValueError("invalid preference pair")
    if algorithm == "simpo":
        if gamma is None or not math.isfinite(gamma) or gamma < 0 or reference is not None:
            raise ValueError("SimPO requires nonnegative gamma and no reference")
        scores = sums / lengths
        margin = beta * (scores[indices[:, 0]] - scores[indices[:, 1]]) - gamma
    elif algorithm == "dpo":
        if gamma is not None or reference is None or reference.shape != sums.shape:
            raise ValueError("DPO requires reference sums and no gamma")
        if not torch.isfinite(reference).all():
            raise ValueError("nonfinite reference score")
        scores = sums - reference.detach()
        margin = beta * (scores[indices[:, 0]] - scores[indices[:, 1]])
    else:
        raise ValueError("unknown algorithm")
    return -F.logsigmoid(margin).mean()


def score_completion(model: Any, prompt: list[int], completion: list[int], chunk_size: int=128) -> Any:
    """Qwen-family decoder; checkpoint vocabulary projection in small chunks."""
    if not prompt or not completion or chunk_size < 1:
        raise ValueError("nonempty prompt/completion and positive chunk size required")
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    device = next(base.parameters()).device
    tokens = torch.tensor([prompt + completion], dtype=torch.long, device=device)
    hidden = base.model(input_ids=tokens[:, :-1], use_cache=False).last_hidden_state
    hidden = hidden[:, len(prompt) - 1:]
    targets = tokens[:, len(prompt):]
    def project(states: Any, labels: Any) -> Any:
        """Compute target-token log probabilities for one hidden-state chunk."""
        logits = base.lm_head(states).float()
        return -F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="sum")
    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, len(completion), chunk_size):
        states, labels = hidden[:, start:start + chunk_size], targets[:, start:start + chunk_size]
        total = total + checkpoint(project, states, labels, use_reentrant=False)
    return total


def group_loss(model: Any, group: dict[str, Any], *, algorithm: str, beta: float, gamma: float | None = None, score: Any=score_completion) -> Any:
    """Score one response view against strict preferences and any frozen reference."""
    responses = group["responses"]
    pairs = strict_pairs(responses)
    if not pairs:
        raise ValueError("group has no preference signal")
    prompt = group["prompt_tokens"]
    reference = None
    if algorithm == "dpo":
        was_training = model.training
        try:
            model.eval()
            with torch.no_grad(), model.disable_adapter():
                reference = torch.stack([score(model, prompt, r["completion_tokens"]) for r in responses])
        finally:
            model.train(was_training)
    sums = torch.stack([score(model, prompt, r["completion_tokens"]) for r in responses])
    lengths = sums.new_tensor([len(r["completion_tokens"]) for r in responses])
    return preference_loss(sums, lengths, pairs, algorithm=algorithm, beta=beta,
                           gamma=gamma, reference=reference)
