"""Evaluate Qwen2.5-1.5B-Instruct on GSM8K test set.

Uses chat template with system prompt asking for \\boxed{} answer.
Batched inference with left-padding.

Usage:
    python scripts/eval_instruct.py
    python scripts/eval_instruct.py --model Qwen/Qwen2.5-Math-1.5B-Instruct
    python scripts/eval_instruct.py --batch_size 32 --limit 200
"""

import argparse
import json
import os
import sys
import time
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import answers_match, extract_number, load_gsm8k_test, set_seed

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "results/gsm8k_instruct"
MAX_NEW_TOKENS = 512
BATCH_SIZE = 512
CHECKPOINT_INTERVAL = 100
SEED = 42

SYSTEM_PROMPT = (
    "You are a helpful math tutor. "
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def build_text(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def run_eval(model, tokenizer, test_data: list, label: str, output_dir: str, batch_size: int) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, f"{label}_results.json")
    metrics_file = os.path.join(output_dir, f"{label}_metrics.json")

    results = []
    if os.path.exists(results_file):
        with open(results_file) as f:
            results = json.load(f)
        print(f"  Resuming: {len(results)}/{len(test_data)} done")

    total_time = 0
    total_tokens = 0
    start_idx = len(results)

    for batch_start in range(start_idx, len(test_data), batch_size):
        batch = test_data[batch_start : batch_start + batch_size]

        texts = [build_text(tokenizer, ex["question"]) for ex in batch]
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        prompt_len = inputs.input_ids.shape[1]

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                do_sample=True,
            )
        elapsed = time.time() - t0
        time_per_q = elapsed / len(batch)
        total_time += elapsed

        for j, ex in enumerate(batch):
            gt_str = ex["ground_truth"]
            try:
                gt_val = float(gt_str.replace(",", "")) if gt_str else None
            except ValueError:
                gt_val = None

            response = tokenizer.decode(outputs[j][prompt_len:], skip_special_tokens=True)
            out_tokens = int(outputs[j].shape[0]) - prompt_len
            total_tokens += out_tokens

            pred_val, extract_method = extract_number(response)
            correct = answers_match(pred_val, gt_val)

            results.append({
                "idx": batch_start + j,
                "question": ex["question"][:200],
                "ground_truth": gt_str,
                "prediction": str(pred_val) if pred_val is not None else None,
                "extract_method": extract_method,
                "correct": correct,
                "response": response,
                "output_tokens": out_tokens,
                "time_s": round(time_per_q, 2),
            })

        done = len(results)
        prev_done = done - len(batch)
        if done // CHECKPOINT_INTERVAL > prev_done // CHECKPOINT_INTERVAL or done >= len(test_data):
            with open(results_file, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=1)
            correct_so_far = sum(1 for r in results if r["correct"])
            tps = total_tokens / total_time if total_time > 0 else 0
            print(
                f"  [{label}] {done}/{len(test_data)} "
                f"acc={correct_so_far/done*100:.1f}% | {tps:.1f} tok/s",
                flush=True,
            )

    with open(results_file, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    total = len(results)
    correct_cnt = sum(1 for r in results if r["correct"])
    truncated = sum(1 for r in results if r["output_tokens"] >= MAX_NEW_TOKENS - 10)
    methods = Counter(r["extract_method"] for r in results)
    peak_vram = torch.cuda.max_memory_allocated() / 1e9

    metrics = {
        "model": label,
        "dataset": "GSM8K test",
        "total": total,
        "max_new_tokens": MAX_NEW_TOKENS,
        "accuracy": round(correct_cnt / total, 4),
        "correct": correct_cnt,
        "truncated": truncated,
        "no_extract": sum(1 for r in results if r["prediction"] is None),
        "extraction_methods": dict(methods),
        "total_time_s": round(total_time, 1),
        "avg_time_s": round(total_time / total, 2) if total > 0 else 0,
        "avg_output_tokens": round(total_tokens / total) if total > 0 else 0,
        "throughput_tps": round(total_tokens / total_time, 1) if total_time > 0 else 0,
        "peak_vram_gb": round(peak_vram, 2),
    }

    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    sep = "=" * 50
    print(f"\n{sep}")
    print(f"  {label}")
    print(f"{sep}")
    print(f"  Accuracy:     {correct_cnt}/{total} ({correct_cnt/total*100:.1f}%)")
    print(f"  Truncated:    {truncated} ({truncated/total*100:.1f}%)")
    print(f"  Extraction:   {dict(methods)}")
    print(f"  Avg tokens:   {total_tokens/total:.0f}")
    print(f"  Peak VRAM:    {peak_vram:.2f} GB")
    print(f"{sep}\n")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only first N examples (for debugging)")
    args = parser.parse_args()

    set_seed(SEED)

    label = args.model.split("/")[-1] + "_cot"

    print(f"Model:   {args.model}")
    print(f"Label:   {label}")
    print(f"System:  {SYSTEM_PROMPT}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()
    print(f"VRAM after load: {torch.cuda.max_memory_allocated()/1e9:.2f} GB\n")

    test_data = load_gsm8k_test()
    if args.limit:
        test_data = test_data[:args.limit]
    print(f"Test set: {len(test_data)} problems\n")

    run_eval(model, tokenizer, test_data, label, args.output_dir, args.batch_size)


if __name__ == "__main__":
    main()
