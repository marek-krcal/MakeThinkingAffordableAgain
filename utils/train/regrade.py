"""Regrade saved GSM8k groups without regenerating responses.

    python -m utils.train.regrade --input groups.jsonl --questions answers.jsonl --output regraded.jsonl

Questions contain id, answer and original prompt_tokens, bound to the source
dataset. Other datasets pass through unchanged; existing outputs are refused.
"""

import argparse
import json
from pathlib import Path

from utils.train.rewards import is_task_correct


def regrade_group(group, questions, tokenizer):
    """Replace only GSM8k labels after verifying gold/prompt identity."""
    if group.get("dataset") != "gsm8k":
        return group
    question = questions[group["id"]]
    if question["prompt_tokens"] != group["prompt_tokens"]:
        raise ValueError(f"gold prompt differs for {group['id']}")
    eos = tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    responses = []
    for response in group["responses"]:
        tokens = response["completion_tokens"]
        if not tokens:
            raise ValueError("empty saved completion")
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        capped = tokens[-1] not in eos_ids
        responses.append(dict(response, correct=is_task_correct(text, question["answer"], "math", capped)))
    return dict(group, responses=responses)


def main():
    """Use a pinned tokenizer and write a separate regraded training input."""
    from transformers import AutoTokenizer
    from evaluate import BASE_MODEL, BASE_REVISION

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "questions", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--tokenizer", default=BASE_MODEL)
    parser.add_argument("--revision", default=BASE_REVISION)
    args = parser.parse_args()
    questions = {}
    for line in args.questions.read_text().splitlines():
        row = json.loads(line)
        if row["id"] in questions:
            raise ValueError("duplicate gold question identity")
        questions[row["id"]] = row
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.revision)
    groups = [json.loads(line) for line in args.input.read_text().splitlines()]
    if len({g["id"] for g in groups}) != len(groups):
        raise ValueError("duplicate group identity")
    regraded = [regrade_group(group, questions, tokenizer) for group in groups]
    with args.output.open("x") as stream:
        stream.writelines(json.dumps(group) + "\n" for group in regraded)


if __name__ == "__main__":
    main()
