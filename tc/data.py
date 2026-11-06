"""Ordered question selection and strict response preferences."""

from __future__ import annotations
from typing import Any

import random


def response_length(response: dict[str, Any]) -> int:
    """Return the saved completion length."""
    tokens = response["completion_tokens"]
    return len(tokens) if isinstance(tokens, list) else tokens


def union(collections: Any) -> list[dict[str, Any]]:
    """Combine rows in input order, rejecting conflicting duplicate identities."""
    rows = {}
    for collection in collections:
        for row in collection:
            identity = row["id"]
            if identity in rows and rows[identity] != row:
                raise ValueError(f"conflicting question ID: {identity}")
            rows[identity] = row
    return list(rows.values())


def select(rows: list[dict[str, Any]], *, split: Any="train", exclude: Any=(), count: int | None = None, seed: int=42) -> list[dict[str, Any]]:
    """Filter before sampling; retain original IDs and seed indices."""
    excluded = set(exclude)
    selected = [r for r in rows if r["split"] == split and r["id"] not in excluded]
    if count is not None:
        if count < 0 or count > len(selected):
            raise ValueError("sample count exceeds available rows")
        indices = sorted(random.Random(seed).sample(range(len(selected)), count))
        selected = [selected[i] for i in indices]
    return selected


def strict_pairs(responses: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Return (chosen, rejected) indices; ungraded responses cannot train."""
    pairs = []
    for i, chosen in enumerate(responses):
        if chosen.get("correct") is not True:
            continue
        for j, rejected in enumerate(responses):
            if rejected.get("correct") is False or (
                rejected.get("correct") is True
                and response_length(chosen) < response_length(rejected)
            ):
                pairs.append((i, j))
    return pairs


def shortest_pairs(responses: list[dict[str, Any]], correct_ratio: float = 1.5) -> list[tuple[int, int]]:
    """Pair the shortest correct answer with longer wrong or sufficiently longer correct answers."""
    correct = [i for i, response in enumerate(responses) if response.get("correct") is True]
    if not correct:
        return []
    chosen = min(correct, key=lambda i: (response_length(responses[i]), responses[i]["sample_index"]))
    length = response_length(responses[chosen])
    return [(chosen, i) for i, response in enumerate(responses)
            if response_length(response) > length and (response.get("correct") is False or
               (response.get("correct") is True and response_length(response) >= correct_ratio * length))]
