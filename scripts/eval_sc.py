"""Self-Consistency evaluation on GSM8K.

Generates N responses per question, takes majority vote as final answer.
Supports SFT and GRPO checkpoints via LoRA adapter loading.

Usage:
    python scripts/eval_sc.py --adapter_path results/grpo/dapo_g16/checkpoint-2760
    python scripts/eval_sc.py --adapter_path results/grpo/dr_grpo_g16/checkpoint-2760 \\
        --sft_adapter results/sft_gsm8k/r32/checkpoint-878 --num_samples 8
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
OUTPUT_DIR = "results/gsm8k_sc"
MAX_NEW_TOKENS = 512
BATCH_SIZE = 8   # вопросов за раз; total sequences = BATCH_SIZE * num_samples
CHECKPOINT_INTERVAL = 100
SEED = 42

PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "Put your final answer within \\boxed{{}}.\n\n"
    "Problem: {question}\n\nSolution:\n"
)


def load_model(base_model: str, adapter_path: str, sft_adapter: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa",
    )
    if sft_adapter:
        model = PeftModel.from_pretrained(model, sft_adapter)
        model = model.merge_and_unload()
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


def majority_vote(preds: list[float | None]) -> tuple[float | None, str]:
    """Return (majority_answer, agreement_str)."""
    valid = [p for p in preds if p is not None]
    if not valid:
        return None, f"0/{len(preds)}"
    rounded = [round(p, 3) for p in valid]
    winner, count = Counter(rounded).most_common(1)[0]
    return winner, f"{count}/{len(preds)}"


def run_eval_sc(
    model, tokenizer, test_data: list, label: str,
    output_dir: str, batch_size: int, num_samples: int,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, f"{label}_sc{num_samples}_results.json")
    metrics_file = os.path.join(output_dir, f"{label}_sc{num_samples}_metrics.json")

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

        # Каждый вопрос повторяется num_samples раз: [q0*N, q1*N, ...]
        texts = [
            PROMPT_TEMPLATE.format(question=ex["question"])
            for ex in batch
            for _ in range(num_samples)
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

        for q_idx, ex in enumerate(batch):
            gt_str = ex["ground_truth"]
            try:
                gt_val = float(gt_str.replace(",", "")) if gt_str else None
            except ValueError:
                gt_val = None

            all_preds, all_responses = [], []
            for s in range(num_samples):
                flat = q_idx * num_samples + s
                resp = tokenizer.decode(outputs[flat][prompt_len:], skip_special_tokens=True)
                pred_val, _ = extract_number(resp)
                all_preds.append(pred_val)
                all_responses.append(resp)
                total_tokens += int(outputs[flat].shape[0]) - prompt_len

            majority, agreement = majority_vote(all_preds)
            correct = answers_match(majority, gt_val)

            results.append({
                "idx": batch_start + q_idx,
                "question": ex["question"][:200],
                "ground_truth": gt_str,
                "prediction": str(majority) if majority is not None else None,
                "correct": correct,
                "agreement": agreement,
                "all_predictions": [str(p) if p is not None else None for p in all_preds],
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
                f"  [{label} SC@{num_samples}] {done}/{len(test_data)} "
                f"acc={correct_so_far/done*100:.1f}% | {tps:.1f} tok/s",
                flush=True,
            )

        del outputs, inputs, texts
        torch.cuda.empty_cache()

    with open(results_file, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    total = len(results)
    correct_cnt = sum(1 for r in results if r["correct"])
    peak_vram = torch.cuda.max_memory_allocated() / 1e9

    # Доля вопросов где majority vote единогласен (все N правильно)
    unanimous = sum(
        1 for r in results
        if r["agreement"].split("/")[0] == str(num_samples)
    )

    metrics = {
        "model": label,
        "dataset": "GSM8K test",
        "num_samples": num_samples,
        "total": total,
        "max_new_tokens": MAX_NEW_TOKENS,
        "accuracy_sc": round(correct_cnt / total, 4),
        "correct": correct_cnt,
        "unanimous_correct": unanimous,
        "total_time_s": round(total_time, 1),
        "avg_time_s": round(total_time / total, 2),
        "peak_vram_gb": round(peak_vram, 2),
    }

    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    sep = "=" * 50
    print(f"\n{sep}")
    print(f"  {label}  SC@{num_samples}")
    print(f"{sep}")
    print(f"  Accuracy SC@{num_samples}: {correct_cnt}/{total} ({correct_cnt/total*100:.1f}%)")
    print(f"  Unanimous correct:        {unanimous}/{total} ({unanimous/total*100:.1f}%)")
    print(f"  Peak VRAM:                {peak_vram:.2f} GB")
    print(f"{sep}\n")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default=BASE_MODEL)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--sft_adapter", default=None,
                        help="Merge SFT adapter before GRPO adapter")
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Questions per batch (total seqs = batch_size * num_samples)")
    parser.add_argument("--num_samples", type=int, default=8)
    args = parser.parse_args()

    set_seed(SEED)

    label = os.path.basename(args.adapter_path)
    total_seqs = args.batch_size * args.num_samples

    print("=" * 60)
    print(f"  Self-Consistency Eval  SC@{args.num_samples}")
    print("=" * 60)
    print(f"  Adapter:        {args.adapter_path}")
    print(f"  SFT adapter:    {args.sft_adapter or 'none'}")
    print(f"  Batch size:     {args.batch_size} questions × {args.num_samples} samples = {total_seqs} seqs")
    print(f"  Output:         {args.output_dir}")
    print("=" * 60)

    model, tokenizer = load_model(args.base_model, args.adapter_path, args.sft_adapter)
    print(f"  VRAM after load: {torch.cuda.max_memory_allocated()/1e9:.2f} GB\n")

    test_data = load_gsm8k_test()
    print(f"Test set: {len(test_data)} problems\n")

    run_eval_sc(model, tokenizer, test_data, label, args.output_dir, args.batch_size, args.num_samples)

    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
