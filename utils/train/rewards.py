"""Checks if the model answered correctly on the suggested datasets.

The model has to end its reply with a line "Final answer: <answer>"; its <think> block is ignored.
Pass the model's replies, the correct answers and the task of each question ("math" for
GSM8K / Skywork, "multiple_choice" for CommonsenseQA / ARC Easy), all as lists of the same length:

    from utils.train.rewards import correctness

    replies = ["<think>...</think>\n\nFinal answer: 42", "<think>...</think>\n\nFinal answer: C"]
    correctness(replies, answer=["42", "B"], task=["math", "multiple_choice"])  # [1.0, 0.0]

For a single reply, is_task_correct(reply, "42", "math") returns True or False.

Math uses the evaluation plain-number and thinking-boundary rules.
Multiple-choice tasks retain their label extraction and formatting tolerance.
"""

import re
from itertools import repeat

from utils.eval.graders import grade_numeric

FINAL_ANSWER = re.compile(r"^Final answer:[ \t]*([^\n]+?)[ \t]*$", re.MULTILINE)
CHOICE_LABEL = re.compile(r"\(?([A-Za-z0-9])\b\)?")
ANSWER_TAG = re.compile(r"</?answer>")
LATEX_WRAPPER = re.compile(r"\\(?:boxed|text)\{(.*)\}")


def correctness(completions, answer, task, truncated=None, **kwargs):
    """Batch of completions -> list of 1.0 (correct) / 0.0 (wrong); usable as a reward function.

    A completion is a string or, as trainers pass it for chat prompts, a list of messages.
    """
    flags = repeat(False) if truncated is None else iter(truncated)
    return [float(is_task_correct(text_of(completion), gold, kind, capped))
            for completion, gold, kind, capped in zip(completions, answer, task, flags)]


def text_of(completion):
    return completion if isinstance(completion, str) else completion[-1]["content"]


def is_task_correct(completion, answer, task, truncated=False):
    """One completion -> True if its final answer matches; task is "math" or "multiple_choice"."""
    if task == "math":
        return grade_numeric(completion, answer, truncated)[0]
    response = completion.split("</think>")[-1]
    predicted = final_answer(response.replace("**", ""))
    if predicted is None:
        return False
    predicted = clean_answer(predicted)
    if task == "multiple_choice":
        return same_choice(predicted, answer)
    raise ValueError(f"Unknown task: {task!r}")


def final_answer(response):
    response = response.strip()
    if "<think>" in response or "</think>" in response:
        return None
    matches = FINAL_ANSWER.findall(response)
    if len(matches) != 1 or not response.splitlines()[-1].startswith("Final answer:"):
        return None
    return matches[0].strip()


def clean_answer(text):
    text = ANSWER_TAG.sub("", text)
    previous = None
    while text != previous:
        previous = text
        latex = LATEX_WRAPPER.fullmatch(text)
        text = latex.group(1) if latex else text
        text = text.strip(" *$<>").rstrip(".")
    return text


def same_choice(predicted, answer):
    label = CHOICE_LABEL.match(predicted)
    return label is not None and label.group(1).upper() == answer.strip().upper()
