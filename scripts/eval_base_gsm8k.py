"""
Evaluate Qwen2.5-1.5B on GSM8K test set (1,319 problems).
Two experiments:
  1. Zero-shot: direct question
  2. CoT: with "Let's think step by step" prompt

Saves full per-problem results + summary metrics for each.
Supports checkpoint/resume.
"""

import json, os, time, torch
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import set_seed, extract_number, answers_match, load_gsm8k_test

MODEL_PATH = "Qwen/Qwen2.5-1.5B"
MAX_NEW_TOKENS = 512
BATCH_SIZE = 512
CHECKPOINT_INTERVAL = 100
OUTPUT_DIR = "results/gsm8k_baseline"


def build_text(prompt: str) -> str:
    # Base model stops on <|endoftext|>, not <|im_end|>.
    # Chat template causes 100% truncation on Qwen2.5-1.5B base.
    return prompt + "\n\n"


# ========== Eval runner ==========

def run_eval(model, tokenizer, test_data, experiment_name, prompt_template):
    results_file = os.path.join(OUTPUT_DIR, f"{experiment_name}_results.json")
    metrics_file = os.path.join(OUTPUT_DIR, f"{experiment_name}_metrics.json")

    results = []
    if os.path.exists(results_file):
        with open(results_file) as f:
            results = json.load(f)
        print(f"  Resuming from checkpoint: {len(results)} already done")

    total_time = 0
    total_tokens = 0
    start_idx = len(results)

    for batch_start in range(start_idx, len(test_data), BATCH_SIZE):
        batch = test_data[batch_start : batch_start + BATCH_SIZE]

        texts = [
            build_text(prompt_template.format(question=ex["question"]))
            for ex in batch
        ]
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

            # With left-padding, all prompts end at prompt_len — slice is correct for every item
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
                f"  [{experiment_name}] {done}/{len(test_data)} "
                f"acc={correct_so_far/done*100:.1f}% | {tps:.1f} tok/s",
                flush=True,
            )

    # Final save
    with open(results_file, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    # Compute metrics
    total = len(results)
    correct_cnt = sum(1 for r in results if r["correct"])
    truncated = sum(1 for r in results if r["output_tokens"] >= MAX_NEW_TOKENS - 10)
    no_extract = sum(1 for r in results if r["prediction"] is None)
    methods = Counter(r["extract_method"] for r in results)
    avg_tokens = total_tokens / total if total > 0 else 0
    peak_vram = torch.cuda.max_memory_allocated() / 1e9

    metrics = {
        "model": "Qwen2.5-1.5B",
        "experiment": experiment_name,
        "dataset": "GSM8K test",
        "total": total,
        "max_new_tokens": MAX_NEW_TOKENS,
        "accuracy": round(correct_cnt / total, 4),
        "correct": correct_cnt,
        "truncated": truncated,
        "no_extract": no_extract,
        "extraction_methods": dict(methods),
        "total_time_s": round(total_time, 1),
        "avg_time_s": round(total_time / total, 2) if total > 0 else 0,
        "avg_output_tokens": round(avg_tokens),
        "throughput_tps": round(total_tokens / total_time, 1) if total_time > 0 else 0,
        "peak_vram_gb": round(peak_vram, 2),
    }

    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  {'='*50}")
    print(f"  {experiment_name}")
    print(f"  {'='*50}")
    print(f"  Accuracy: {correct_cnt}/{total} ({correct_cnt/total*100:.1f}%)")
    print(f"  Truncated: {truncated}")
    print(f"  No extract: {no_extract}")
    print(f"  Extraction: {dict(methods)}")
    print(f"  Avg time: {total_time/total:.2f}s/q")
    print(f"  Throughput: {total_tokens/total_time:.1f} tok/s")
    print(f"  Peak VRAM: {peak_vram:.2f} GB")
    print(f"  {'='*50}\n")

    return metrics

# ========== Main ==========

SEED = 42

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_seed(SEED)

    print("Loading Qwen2.5-1.5B...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for batched generation
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map="auto"
    )
    print(f"VRAM after load: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    test_data = load_gsm8k_test()
    print(f"Test set: {len(test_data)} problems\n")

    # ===== Experiment 1: Zero-shot =====
    print("=" * 60)
    print("  Experiment 1: Zero-shot (direct question)")
    print("=" * 60)
    zeroshot_prompt = "{question}"
    run_eval(model, tokenizer, test_data, "base_zeroshot_test3", zeroshot_prompt)

    # ===== Experiment 2: CoT prompting =====
    print("=" * 60)
    print("  Experiment 2: CoT (Let's think step by step)")
    print("=" * 60)
    cot_prompt = "{question}\n\nLet's think step by step and put the final answer after ####."
    run_eval(model, tokenizer, test_data, "base_cot_test3", cot_prompt)

    print("\nAll experiments complete!")

if __name__ == "__main__":
    main()
