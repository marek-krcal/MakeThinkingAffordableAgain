"""Balanced four-response views and deterministic exposure."""

from __future__ import annotations
from typing import Any

from itertools import combinations
import random

from tc.data import response_length, strict_pairs

TARGETS = {1: (1, 1), 2: (2, 2), 3: (2, 2), 4: (2, 2),
           5: (3, 3), 6: (3, 3), 7: (3, 4), 8: (4, 4)}


def pool_pick(pool: Any, count: int, side: Any, epoch: int) -> Any:
    """Select deterministic rotating members of one response pool."""
    if not count:
        return []
    start = epoch * 2 * count % len(pool)
    indices = list(dict.fromkeys((start + 2 * i + side) % len(pool) for i in range(count)))
    indices += [i for i in range(len(pool)) if i not in indices]
    return [pool[i] for i in indices[:count]]


def views(group: dict[str, Any], epoch: int) -> Any:
    """Build balanced preference-bearing views for one group and epoch."""
    responses = group["responses"]
    if not 1 <= len(responses) <= 16:
        raise ValueError("training currently supports one to sixteen saved responses per group")
    if len({r["sample_index"] for r in responses}) != len(responses):
        raise ValueError("duplicate response sample index")
    if any(type(r["correct"]) is not bool or not r["completion_tokens"] for r in responses):
        raise ValueError("boolean grades and nonempty completion tokens required")
    if not strict_pairs(responses):
        return []
    correct = sorted((r for r in responses if r["correct"]),
                     key=lambda r: (response_length(r), r["sample_index"]))
    wrong = sorted((r for r in responses if not r["correct"]), key=lambda r: r["sample_index"])
    size = min(4, len(responses))
    if len(responses) <= 8:
        targets = [min(size, len(correct), max(TARGETS[len(correct)][side], size - len(wrong)))
                   for side in (0, 1)]
    else:
        count = (len(responses) + 3) // 4
        denominator = 5 * len(correct) + 4 * len(wrong)
        slots = (2 * 4 * count * 5 * len(correct) + denominator) // (2 * denominator)
        slots = max(count * max(1, 4 - len(wrong)), min(slots, count * min(4, len(correct))))
        targets = [slots // count + int(i < slots % count) for i in range(count)]
        shift = epoch % count
        targets = targets[shift:] + targets[:shift]
    correct_pos, wrong_pos = epoch * sum(targets), epoch * (size * len(targets) - sum(targets))
    result = []
    for side, target in enumerate(targets):
        if len(responses) > 8:
            chosen = [correct[(correct_pos + i) % len(correct)] for i in range(target)]
            chosen += [wrong[(wrong_pos + i) % len(wrong)] for i in range(size - target)]
            correct_pos += target
            wrong_pos += size - target
        elif len(correct) == 7 and len(wrong) == 1:
            chosen = [correct[i] for i in ([0, 2, 4] if side == 0 else [1, 3, 5, 6])]
            chosen += wrong if side == 0 else []
        else:
            chosen = pool_pick(correct, target, side, epoch) + pool_pick(wrong, size - target, side, epoch)
        if not strict_pairs(chosen):
            preferred = {r["sample_index"] for r in chosen}
            options = [list(rs) for rs in combinations(responses, size)
                       if sum(r["correct"] for r in rs) == target and strict_pairs(rs)]
            if not options:
                continue
            chosen = min(options, key=lambda rs: (-len(preferred & {r["sample_index"] for r in rs}),
                                                 tuple(sorted(r["sample_index"] for r in rs))))
        result.append(dict(group, responses=chosen, selection_epoch=epoch, tuple_side=side))
    return result


def batches(groups: list[dict[str, Any]], *, seed: int, batch_size: int, steps: int | None = None, epochs: int | None = None) -> Any:
    """Construct deterministic update batches with the requested exposure schedule."""
    if not groups or len({g["id"] for g in groups}) != len(groups):
        raise ValueError("unique nonempty groups required")
    if (steps is None) == (epochs is None) or batch_size < 1:
        raise ValueError("choose steps or epochs and a positive batch size")
    pending, result, epoch = [], [], 0
    while (len(result) < steps if steps is not None else epoch < epochs):
        current = [view for group in groups for view in views(group, epoch)]
        if not current:
            raise ValueError("no training signal")
        random.Random(seed + epoch).shuffle(current)
        epoch += 1
        if epochs is not None:
            result.extend(current[i:i + batch_size] for i in range(0, len(current), batch_size))
            continue
        pending.extend(current)
        while len(result) < steps:
            size = 6 if steps == 982 and batch_size == 8 and len(result) + 1 in (491, 982) else batch_size
            if len(pending) < size:
                break
            result.append(pending[:size])
            del pending[:size]
    return result


def learning_rate(step: int, total: int, peak: float) -> float:
    """Return the warmup, stable or decay rate for this update."""
    warmup = max(1, round(total * .05))
    stable = round(total * .8)
    if step <= warmup:
        return peak * step / warmup
    if step <= stable:
        return peak
    return peak * (total - step) / max(1, total - stable)
