"""
Prepare SFT data in Plan B format from raw OpenMathInstruct-2.

Pipeline (single pass, no intermediate files):
  data/raw/OpenMathInstruct-2/augmented_gsm8k.jsonl
      → group by problem, keep shortest correct solution
      → format as prompt-completion with <|endoftext|>
      → data/processed_gsm8k_sft_planb/{sft_train,sft_val}.jsonl

Usage:
    python scripts/prepare_sft_data.py
"""

import json
import os
import random
import sys
from collections import defaultdict

from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import extract_boxed

MODEL_PATH = "Qwen/Qwen2.5-1.5B"
RAW_PATH = "data/raw/OpenMathInstruct-2/augmented_gsm8k.jsonl"
OUT_DIR = "data/processed_gsm8k_sft_planb"
VAL_SIZE = 3000
TRAIN_SIZE = 67000
SEED = 42

PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "Put your final answer within \\boxed{{}}.\n\n"
    "Problem: {question}\n\nSolution:\n"
)


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    eos = tokenizer.eos_token  # <|endoftext|> for Qwen2.5 base
    print(f"EOS token: {repr(eos)} (id={tokenizer.eos_token_id})")

    # ── 1. Load raw data ──────────────────────────────────────────────────
    print(f"\nLoading {RAW_PATH}...")
    raw = []
    with open(RAW_PATH) as f:
        for line in f:
            raw.append(json.loads(line))
    print(f"  Total raw examples: {len(raw)}")

    # ── 2. Group by problem, keep shortest correct solution ───────────────
    by_problem = defaultdict(list)
    for ex in raw:
        by_problem[ex["problem"]].append(ex)
    print(f"  Unique problems: {len(by_problem)}")

    candidates = []
    skipped = 0
    for solutions in by_problem.values():
        solutions.sort(key=lambda x: len(x["generated_solution"]))
        for sol in solutions:
            boxed = extract_boxed(sol["generated_solution"])
            if boxed is None:
                continue
            try:
                pred = float(boxed.replace(",", ""))
                gt = float(sol["expected_answer"].replace(",", ""))
                match = abs(pred - gt) < 1e-6
            except (ValueError, TypeError):
                match = boxed.strip() == sol["expected_answer"].strip()
            if match:
                candidates.append(sol)
                break
        else:
            skipped += 1

    print(f"  Valid candidates: {len(candidates)}  (skipped {skipped} with no correct \\boxed{{}})")

    # ── 3. Shuffle and split ──────────────────────────────────────────────
    random.seed(SEED)
    random.shuffle(candidates)
    val_data = candidates[:VAL_SIZE]
    train_data = candidates[VAL_SIZE : VAL_SIZE + TRAIN_SIZE]
    print(f"  Split → train: {len(train_data)}, val: {len(val_data)}")

    # ── 4. Format as Plan B and save ──────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)

    for split_name, split_data in [("sft_train", train_data), ("sft_val", val_data)]:
        out_path = os.path.join(OUT_DIR, f"{split_name}.jsonl")
        records = []
        for ex in split_data:
            prompt = PROMPT_TEMPLATE.format(question=ex["problem"])
            completion = ex["generated_solution"] + eos
            records.append({
                "prompt": prompt,
                "completion": completion,
                "expected_answer": ex["expected_answer"],
            })

        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Token length stats (on first 2000 examples for speed)
        sample = records[:2000]
        full_lens = sorted(len(tokenizer.encode(r["prompt"] + r["completion"])) for r in sample)
        comp_lens = sorted(len(tokenizer.encode(r["completion"])) for r in sample)
        n = len(full_lens)
        print(f"\n  {split_name}: {len(records)} examples → {out_path}")
        print(f"    Full (prompt+completion): mean={sum(full_lens)/n:.0f}  P95={full_lens[int(n*0.95)]}  max={full_lens[-1]}")
        print(f"    Completion only:          mean={sum(comp_lens)/n:.0f}  P95={comp_lens[int(n*0.95)]}  max={comp_lens[-1]}")
        print(f"    Ends with EOS: {records[0]['completion'].endswith(eos)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
