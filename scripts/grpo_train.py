"""GRPO training on GSM8K starting from SFT checkpoint.

Uses TRL GRPOTrainer with Dr. GRPO loss (decoupled reward normalization).
Pre-filtered training data via filter_grpo.py (Dynamic Sampling).

Usage:
    python scripts/grpo_train.py --no_wandb
    python scripts/grpo_train.py --adapter_path results/sft_gsm8k/r32/checkpoint-878
"""

import argparse
import json
import os
import sys
import time

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOTrainer, GRPOConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import extract_boxed, set_seed

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-1.5B"
DEFAULT_ADAPTER_PATH = os.path.join(PROJECT_ROOT, "results", "sft_gsm8k", "r32", "final")
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "grpo_filtered", "grpo_train.jsonl")

PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "Put your final answer within \\boxed{{}}.\n\n"
    "Problem: {question}\n\nSolution:\n"
)


class RealtimeLogCallback(TrainerCallback):
    """Write GRPO metrics to CSV in real-time."""

    def __init__(self, log_path: str):
        self.f = open(log_path, "a")
        self.f.write("step,epoch,loss,reward_mean,reward_std,frac_zero_std,completion_len,clip_ratio,entropy,lr\n")
        self.f.flush()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        row = [
            state.global_step,
            round(state.epoch, 4) if state.epoch else "",
            logs.get("loss", ""),
            logs.get("reward/gsm8k_reward/mean", logs.get("reward", "")),
            logs.get("reward/gsm8k_reward/std", logs.get("reward_std", "")),
            logs.get("frac_reward_zero_std", ""),
            logs.get("completions/mean_length", ""),
            logs.get("clip_ratio/region_mean", ""),
            logs.get("entropy", ""),
            logs.get("learning_rate", ""),
        ]
        self.f.write(",".join(str(v) for v in row) + "\n")
        self.f.flush()

    def on_train_end(self, **_):
        self.f.close()


def gsm8k_reward(completions, ground_truth, **kwargs) -> list[float]:
    """Graded reward: +1.0 correct, 0.0 wrong format but has \\boxed{}, -0.5 no \\boxed{}."""
    rewards = []
    for completion, gt in zip(completions, ground_truth):
        if isinstance(completion, list):
            text = completion[-1]["content"] if completion else ""
        elif isinstance(completion, dict):
            text = completion.get("content", "")
        else:
            text = str(completion)

        pred = extract_boxed(text)
        if pred is None:
            rewards.append(-0.5)
            continue

        try:
            correct = abs(float(pred.replace(",", "")) - float(gt.replace(",", ""))) < 1e-3
        except (ValueError, TypeError):
            correct = pred.strip() == str(gt).strip()

        rewards.append(1.0 if correct else 0.0)

    return rewards


def main():
    parser = argparse.ArgumentParser(description="GRPO Training on GSM8K")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--adapter_path", type=str, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--loss_type", type=str, default="dr_grpo",
                        choices=["grpo", "dapo", "dr_grpo"],
                        help="GRPO loss variant")
    parser.add_argument("--num_generations", type=int, default=16)
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    if args.output_dir is None:
        args.output_dir = os.path.join(PROJECT_ROOT, "results", "grpo", f"{args.loss_type}_g{args.num_generations}")

    print("=" * 60)
    print(f"  GRPO Training ({args.loss_type.upper()})")
    print("=" * 60)
    print(f"  Model:          {args.model_path}")
    print(f"  SFT adapter:    {args.adapter_path}")
    print(f"  Data:           {args.data_path}")
    print(f"  Loss type:      {args.loss_type}")
    print(f"  Generations:    {args.num_generations} (batch={args.batch_size})")
    print(f"  Max completion: {args.max_completion_length}")
    print(f"  Epochs:         {args.num_epochs}")
    print(f"  Eff. batch:     {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"  LR:             {args.lr}")
    print(f"  Output:         {args.output_dir}")
    print("=" * 60)

    print(f"\nLoading data from {args.data_path}...")
    with open(args.data_path) as f:
        grpo_data = [json.loads(line) for line in f]
    train_dataset = Dataset.from_list(grpo_data)
    print(f"  Training set: {len(train_dataset)} questions")

    steps_per_epoch = len(train_dataset) // (args.batch_size * args.grad_accum)
    warmup_steps = max(1, round(0.1 * steps_per_epoch * args.num_epochs))
    print(f"  Warmup steps: {warmup_steps}")

    print("\nLoading model + SFT adapter...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model = model.merge_and_unload()
    print(f"  VRAM after load: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 4,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
\
    os.makedirs(args.output_dir, exist_ok=True)
    log_csv_path = os.path.join(args.output_dir, f"train_log_{args.loss_type}.csv")
    log_callback = RealtimeLogCallback(log_csv_path)

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        loss_type=args.loss_type,
        beta=0.0,
        epsilon=0.2,
        epsilon_high=0.28,
        mask_truncated_completions=True,
        num_generations=args.num_generations,
        generation_batch_size=args.batch_size * args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=0.7,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        bf16=True,
        use_vllm=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=4,
        seed=args.seed,
        report_to="none" if args.no_wandb else "wandb",
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        reward_funcs=[gsm8k_reward],
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=[log_callback],
    )

    print(f"  Loss log: {log_csv_path}")
    print(f"\nStarting {args.loss_type.upper()} training...")
    t_start = time.time()
    train_result = trainer.train()
    t_elapsed = time.time() - t_start

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    summary = {
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "data_path": args.data_path,
        "loss_type": args.loss_type,
        "beta": 0.0,
        "epsilon": 0.2,
        "epsilon_high": 0.28,
        "mask_truncated_completions": True,
        "num_generations": args.num_generations,
        "max_completion_length": args.max_completion_length,
        "temperature": 0.7,
        "num_epochs": args.num_epochs,
        "per_device_train_batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.lr,
        "train_runtime_min": round(t_elapsed / 60, 1),
        "peak_vram_gb": round(peak_vram, 2),
        "total_steps": trainer.state.global_step,
    }

    with open(os.path.join(args.output_dir, f"training_summary_{args.loss_type}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output_dir, f"log_history_{args.loss_type}.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  {args.loss_type.upper()} Training Complete!")
    print(f"{'=' * 60}")
    print(f"  Training time: {round(t_elapsed / 60, 1)} min")
    print(f"  Peak VRAM:     {round(peak_vram, 2)} GB")
    print(f"  Saved to:      {final_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
