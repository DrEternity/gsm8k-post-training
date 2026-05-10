"""Evaluate Qwen2.5-1.5B SFT checkpoints on GSM8K test set.

Automatically discovers all checkpoints in CHECKPOINTS_DIR and evaluates each.
Uses batched inference for speed.

Usage:
    python scripts/eval_sft.py
    python scripts/eval_sft.py --checkpoints_dir results/sft_gsm8k/r32 --batch_size 32
    python scripts/eval_sft.py --adapter_path results/sft_gsm8k/r32/checkpoint-439
"""

import argparse
import gc
import json
import os
import sys
import time
from collections import Counter

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import answers_match, extract_number, load_gsm8k_test, set_seed

BASE_MODEL = "Qwen/Qwen2.5-1.5B"
CHECKPOINTS_DIR = "results/sft_gsm8k/r32"
OUTPUT_DIR = "results/gsm8k_sft"
MAX_NEW_TOKENS = 512
BATCH_SIZE = 512
CHECKPOINT_INTERVAL = 100
SEED = 42

PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "Put your final answer within \\boxed{{}}.\n\n"
    "Problem: {question}\n\nSolution:\n"
)


def find_checkpoints(checkpoints_dir: str) -> list[str]:
    """Return all checkpoint dirs sorted by step number."""
    if not os.path.exists(checkpoints_dir):
        return []
    entries = []
    for name in os.listdir(checkpoints_dir):
        if name.startswith("checkpoint-"):
            try:
                step = int(name.split("-")[1])
                entries.append((step, os.path.join(checkpoints_dir, name)))
            except ValueError:
                pass
    entries.sort()
    return [path for _, path in entries]


def load_model(base_model: str, adapter_path: str, sft_adapter: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16, device_map="auto"
    )
    if sft_adapter:
        model = PeftModel.from_pretrained(model, sft_adapter)
        model = model.merge_and_unload()
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


def run_eval(model, tokenizer, test_data: list, label: str, output_dir: str, batch_size: int = BATCH_SIZE) -> dict:
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

        texts = [PROMPT_TEMPLATE.format(question=ex["question"]) for ex in batch]
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
        "model": f"Qwen2.5-1.5B + SFT ({label})",
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
    print(f"  Truncated:    {truncated}")
    print(f"  Extraction:   {dict(methods)}")
    print(f"  Avg tokens:   {total_tokens/total:.0f}")
    print(f"  Peak VRAM:    {peak_vram:.2f} GB")
    print(f"{sep}\n")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default=BASE_MODEL)
    parser.add_argument("--checkpoints_dir", default=CHECKPOINTS_DIR,
                        help="Directory with checkpoint-* subdirs to evaluate all of them")
    parser.add_argument("--adapter_path", default=None,
                        help="Evaluate a single adapter instead of all checkpoints")
    parser.add_argument("--sft_adapter", default=None,
                        help="SFT adapter to merge before applying the main adapter (for GRPO eval)")
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    set_seed(SEED)

    # Determine which adapters to evaluate
    if args.adapter_path:
        adapters = [args.adapter_path]
    else:
        adapters = find_checkpoints(args.checkpoints_dir)
        if not adapters:
            print(f"No checkpoints found in {args.checkpoints_dir}")
            return

    print(f"Found {len(adapters)} checkpoint(s) to evaluate:")
    for p in adapters:
        print(f"  {p}")

    test_data = load_gsm8k_test()
    print(f"\nTest set: {len(test_data)} problems\n")

    all_metrics = []
    for adapter_path in adapters:
        label = os.path.basename(adapter_path)
        print("=" * 60)
        print(f"  Evaluating: {label}")
        print("=" * 60)

        model, tokenizer = load_model(args.base_model, adapter_path, args.sft_adapter)
        print(f"  VRAM after load: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

        metrics = run_eval(model, tokenizer, test_data, label, args.output_dir, args.batch_size)
        metrics["adapter_path"] = adapter_path
        all_metrics.append(metrics)

        # Free GPU memory before next checkpoint
        del model
        torch.cuda.empty_cache()
        gc.collect()

    # Summary across all checkpoints
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  {'Checkpoint':<25} {'Accuracy':>10} {'Truncated':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10}")
    for m in all_metrics:
        label = os.path.basename(m["adapter_path"])
        print(f"  {label:<25} {m['accuracy']*100:>9.1f}% {m['truncated']:>10}")

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
