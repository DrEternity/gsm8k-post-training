"""Filter GSM8K train set for GRPO training.

For each question generates NUM_ROLLOUTS completions and computes success_rate.
Filters out only Easy questions (rate=1.0) — model already solves them reliably.

Kept categories:
  - Medium (0 < rate < 1): model sometimes solves → clear reward signal
  - Hard   (rate = 0.0):   model never solves now, but may after a few GRPO steps

Usage:
    python scripts/filter_grpo.py --adapter_path results/sft_gsm8k/r32/checkpoint-439
"""

import argparse
import json
import os
import sys
import time

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import answers_match, extract_boxed, extract_gsm8k_answer, set_seed

BASE_MODEL = "Qwen/Qwen2.5-1.5B"
DEFAULT_ADAPTER = "results/sft_gsm8k/r32/final"
OUTPUT_DIR = "data/grpo_filtered"
NUM_ROLLOUTS = 8
QUESTION_BATCH = 100
MAX_NEW_TOKENS = 1024
CHECKPOINT_INTERVAL = 100
SEED = 42

PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "Put your final answer within \\boxed{{}}.\n\n"
    "Problem: {question}\n\nSolution:\n"
)


def load_gsm8k_train() -> list[dict]:
    ds = load_dataset("openai/gsm8k", "main", split="train")
    return [
        {"question": ex["question"], "ground_truth": extract_gsm8k_answer(ex["answer"])}
        for ex in ds
    ]


def is_correct(response: str, gt_str: str | None) -> bool:
    """Check correctness strictly via \\boxed{} — no fallback to last_num."""
    if gt_str is None:
        return False
    pred_str = extract_boxed(response)
    try:
        pred_val = float(pred_str.replace(",", "")) if pred_str else None
        gt_val = float(gt_str.replace(",", ""))
    except (ValueError, TypeError):
        return False
    return answers_match(pred_val, gt_val)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default=BASE_MODEL)
    parser.add_argument("--adapter_path", default=DEFAULT_ADAPTER)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--num_rollouts", type=int, default=NUM_ROLLOUTS)
    parser.add_argument("--question_batch", type=int, default=QUESTION_BATCH)
    args = parser.parse_args()

    set_seed(SEED)
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "filter_results.json")

    print(f"Base model:  {args.base_model}")
    print(f"Adapter:     {args.adapter_path}")
    print(f"Rollouts:    {args.num_rollouts}  |  Question batch: {args.question_batch}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model = model.merge_and_unload()
    model.eval()
    print(f"VRAM after load: {torch.cuda.max_memory_allocated()/1e9:.2f} GB\n")

    print("Loading GSM8K train set...")
    train_data = load_gsm8k_train()
    print(f"Total questions: {len(train_data)}")

    # Resume from checkpoint
    results = []
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        print(f"Resuming from {len(results)} done")

    start_idx = len(results)
    t0 = time.time()

    for batch_start in range(start_idx, len(train_data), args.question_batch):
        batch = train_data[batch_start : batch_start + args.question_batch]

        all_prompts = [
            PROMPT_TEMPLATE.format(question=ex["question"])
            for ex in batch
            for _ in range(args.num_rollouts)
        ]
        inputs = tokenizer(all_prompts, return_tensors="pt", padding=True).to(model.device)
        prompt_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                do_sample=True,
            )

        for q_idx, ex in enumerate(batch):
            rollouts = []
            correct_count = 0
            for r in range(args.num_rollouts):
                flat_idx = q_idx * args.num_rollouts + r
                response = tokenizer.decode(outputs[flat_idx][prompt_len:], skip_special_tokens=True)
                correct = is_correct(response, ex["ground_truth"])
                if correct:
                    correct_count += 1
                rollouts.append({"response": response, "correct": correct})

            results.append({
                "idx": batch_start + q_idx,
                "question": ex["question"],
                "ground_truth": ex["ground_truth"],
                "success_rate": correct_count / args.num_rollouts,
                "correct_count": correct_count,
                "num_rollouts": args.num_rollouts,
                "rollouts": rollouts,
            })

        del outputs, inputs, all_prompts
        torch.cuda.empty_cache()

        done = len(results)
        prev_done = done - len(batch)
        if done // CHECKPOINT_INTERVAL > prev_done // CHECKPOINT_INTERVAL or done >= len(train_data):
            with open(results_path, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=1)
            rates = [r["success_rate"] for r in results]
            easy = sum(1 for r in rates if r == 1.0)
            hard = sum(1 for r in rates if r == 0.0)
            medium = done - easy - hard
            elapsed = time.time() - t0
            print(
                f"  [{done}/{len(train_data)}] "
                f"easy={easy}({easy/done*100:.0f}%) "
                f"medium={medium}({medium/done*100:.0f}%) "
                f"hard={hard}({hard/done*100:.0f}%) | "
                f"{elapsed/done:.2f}s/q",
                flush=True,
            )

    with open(results_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    # Stats
    total = len(results)
    rates = [r["success_rate"] for r in results]
    easy = sum(1 for r in rates if r == 1.0)
    hard = sum(1 for r in rates if r == 0.0)
    medium = total - easy - hard

    print(f"\n{'='*50}")
    print(f"  Filtering Results")
    print(f"{'='*50}")
    print(f"  Total:              {total}")
    print(f"  Easy  (rate=1.0):   {easy:>5} ({easy/total*100:.1f}%)")
    print(f"  Medium (0<rate<1):  {medium:>5} ({medium/total*100:.1f}%)")
    print(f"  Hard  (rate=0.0):   {hard:>5} ({hard/total*100:.1f}%)")

    # Distribution breakdown
    buckets: dict[str, int] = {}
    for r in rates:
        key = f"{int(r * args.num_rollouts)}/{args.num_rollouts}"
        buckets[key] = buckets.get(key, 0) + 1
    print("\n  Success rate distribution:")
    for k in sorted(buckets, key=lambda x: int(x.split("/")[0])):
        v = buckets[k]
        print(f"    {k}: {v:>5} ({v/total*100:.1f}%)")

    # Save: keep all except easy (success_rate < 1.0)
    grpo_data = [
        {
            "prompt": PROMPT_TEMPLATE.format(question=r["question"]),
            "ground_truth": r["ground_truth"],
            "success_rate": r["success_rate"],
        }
        for r in results
        if r["success_rate"] < 1.0
    ]

    grpo_path = os.path.join(args.output_dir, "grpo_train.jsonl")
    with open(grpo_path, "w") as f:
        for d in grpo_data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"\n  GRPO training set (medium + hard): {len(grpo_data)} questions")
    print(f"  Saved to: {grpo_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
