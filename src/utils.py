"""Utility functions: seeding, answer extraction, GSM8K data loading."""

import json
import os
import random
import re
from collections import Counter

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_boxed(text: str) -> str | None:
    r"""Extract raw string content from the last \boxed{...} in text.

    Uses brace-depth tracking to handle nested braces correctly.
    Returns the raw string (e.g. "42", "\\frac{1}{2}"), not a float —
    so it's usable both in reward functions and in numeric eval pipelines.
    """
    marker = r"\boxed{"
    idx = text.rfind(marker)
    if idx == -1:
        return None
    start = idx + len(marker)
    depth, i = 1, start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start : i - 1].strip() if depth == 0 else None


def extract_gsm8k_answer(text: str) -> str | None:
    """Extract the numeric answer after #### in GSM8K ground-truth format."""
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    return None


def extract_number(text: str) -> tuple[float | None, str]:
    r"""Cascade extraction: \boxed{} → #### → last number in text.

    Returns (value_as_float_or_None, method_string).
    method_string is one of: "boxed", "####", "last_num", "no_extract".
    """
    boxed = extract_boxed(text)
    if boxed is not None:
        try:
            return float(boxed.replace(",", "")), "boxed"
        except ValueError:
            pass

    gsm = extract_gsm8k_answer(text)
    if gsm is not None:
        try:
            return float(gsm), "####"
        except ValueError:
            pass

    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        try:
            return float(numbers[-1]), "last_num"
        except ValueError:
            pass

    return None, "no_extract"


def answers_match(pred: float | None, gt: float | None) -> bool:
    """Return True when pred and gt are within 1e-3 of each other."""
    if pred is None or gt is None:
        return False
    return abs(pred - gt) < 1e-3


# ---------------------------------------------------------------------------
# GSM8K data loading
# ---------------------------------------------------------------------------

def load_gsm8k_test(test_path: str = "data/processed/sft_test.jsonl") -> list[dict]:
    """Load GSM8K test set.

    Tries a local preprocessed JSONL first; falls back to HuggingFace Hub.
    Each returned dict has keys: "question" (str) and "ground_truth" (str).
    """
    if os.path.exists(test_path):
        print(f"Loading GSM8K test from {test_path}...")
        data = []
        with open(test_path) as f:
            for line in f:
                d = json.loads(line)
                question = d["messages"][1]["content"]
                answer_text = d["messages"][2]["content"]
                gt = extract_gsm8k_answer(answer_text)
                if gt is None:
                    nums = re.findall(r"-?\d+\.?\d*", answer_text)
                    gt = nums[-1] if nums else None
                data.append({"question": question, "ground_truth": gt})
        return data

    print("Loading GSM8K test from HuggingFace Hub...")
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    return [
        {"question": ex["question"], "ground_truth": extract_gsm8k_answer(ex["answer"])}
        for ex in ds
    ]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    predictions: list[float | None],
    ground_truths: list[float | None],
    formats: list[str] | None = None,
) -> dict:
    """Compute accuracy and extraction-format breakdown."""
    if not predictions:
        return {"accuracy": 0.0, "extracted": 0.0}

    total = len(predictions)
    correct = sum(
        1
        for pred, gt in zip(predictions, ground_truths)
        if answers_match(pred, gt)
    )
    result = {
        "accuracy": correct / total,
        "extracted": sum(1 for p in predictions if p is not None) / total,
    }
    if formats:
        fmt_counts = Counter(formats)
        result["format_breakdown"] = {k: fmt_counts.get(k, 0) / total for k in ("boxed", "####", "last_num", "no_extract")}
    return result


# ---------------------------------------------------------------------------
# Legacy aliases (kept for backward compatibility with reward.py)
# ---------------------------------------------------------------------------

def extract_final_answer_qwen3(response: str) -> tuple[float | None, str]:
    """Cascade extraction with </think> stripping. Alias for reward.py."""
    if "</think>" in response:
        response = response.split("</think>")[-1]
    return extract_number(response)
