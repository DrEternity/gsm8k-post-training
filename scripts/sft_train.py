"""LoRA SFT training script for Qwen2.5-1.5B on GSM8K.

Uses SFTConfig completion_only_loss=True to only compute loss on completion part.
Supports resume_from_checkpoint. Logs loss to file in real-time.
"""

import argparse
import json
import os
import sys
import time

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTTrainer, SFTConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import set_seed

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-1.5B"
DEFAULT_TRAIN_DATA = os.path.join(PROJECT_ROOT, "data", "processed_gsm8k_sft_planb", "sft_train.jsonl")
DEFAULT_VAL_DATA = os.path.join(PROJECT_ROOT, "data", "processed_gsm8k_sft_planb", "sft_val.jsonl")


class LoggingCallback(TrainerCallback):
    """Write loss and metrics to a log file in real-time."""

    def __init__(self, log_path):
        self.log_path = log_path
        self.f = open(log_path, "a")
        self.f.write("step,epoch,loss,learning_rate,grad_norm,eval_loss,tokens_per_second\n")
        self.f.flush()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        epoch = round(state.epoch, 4) if state.epoch else ""
        loss = logs.get("loss", "")
        lr = logs.get("learning_rate", "")
        grad_norm = logs.get("grad_norm", "")
        eval_loss = logs.get("eval_loss", "")
        tps = logs.get("tokens_per_second", "")
        self.f.write(f"{step},{epoch},{loss},{lr},{grad_norm},{eval_loss},{tps}\n")
        self.f.flush()

    def on_train_end(self, **_):
        self.f.close()


def load_jsonl(path: str) -> Dataset:
    with open(path) as f:
        data = [json.loads(line) for line in f]
    return Dataset.from_list(data)


def main():
    parser = argparse.ArgumentParser(description="SFT Training")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--train_data", type=str, default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--val_data", type=str, default=DEFAULT_VAL_DATA)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint directory to resume from")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.lora_alpha is None:
        args.lora_alpha = args.lora_r * 4
    if args.output_dir is None:
        args.output_dir = os.path.join(PROJECT_ROOT, "results", "sft_gsm8k", f"r{args.lora_r}")

    print("=" * 60)
    print("  SFT Training - Qwen2.5-1.5B on GSM8K")
    print("=" * 60)
    print(f"  Model:         {args.model_path}")
    print(f"  Method:        LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"  Epochs:        {args.num_epochs}")
    print(f"  Batch size:    {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print(f"  LR:            {args.lr}")
    print(f"  Max seq len:   {args.max_seq_length}")
    print(f"  Loss:          completion_only")
    print(f"  Resume from:   {args.resume_from or 'None (fresh start)'}")
    print(f"  Output:        {args.output_dir}")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    train_data = load_jsonl(args.train_data)
    val_data = load_jsonl(args.val_data)
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")
    steps_per_epoch = len(train_data) // (args.batch_size * args.grad_accum)
    warmup_steps = 100
    print(f"  Warmup steps: {warmup_steps} (5% of {steps_per_epoch * args.num_epochs} total)")

    # Load model
    print(f"\nLoading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    print(f"  eos_token: {repr(tokenizer.eos_token)} (id={tokenizer.eos_token_id})")
    print(f"  VRAM after load: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type=TaskType.CAUSAL_LM,
    )

    # Real-time loss logging
    os.makedirs(args.output_dir, exist_ok=True)
    log_csv_path = os.path.join(args.output_dir, "train_log.csv")
    logging_callback = LoggingCallback(log_csv_path)

    # Training config
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size // 2),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        max_length=args.max_seq_length,
        completion_only_loss=True,
        bf16=True,
        padding_free=True,
        packing=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        logging_first_step=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=6,
        seed=args.seed,
        report_to="none" if args.no_wandb else "wandb",
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=lora_config,
        callbacks=[logging_callback],
    )

    # LoRA applied by SFTTrainer — count after init
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n  Trainable params: {trainable / 1e6:.1f}M / {total / 1e9:.2f}B ({trainable/total*100:.2f}%)")
    print(f"  Loss log: {log_csv_path}")

    # Train (with optional resume)
    print("\nStarting training...")
    t_start = time.time()
    train_result = trainer.train(resume_from_checkpoint=args.resume_from)
    t_elapsed = time.time() - t_start

    # Save
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nModel saved to {final_dir}")

    # Save summaries
    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    summary = {
        "gpu": torch.cuda.get_device_name(0),
        "model_path": args.model_path,
        "method": "lora_sft",
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "trainable_params_M": round(trainable / 1e6, 1),
        "total_params_B": round(total / 1e9, 2),
        "trainable_pct": round(trainable / total * 100, 2),
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "learning_rate": args.lr,
        "max_seq_length": args.max_seq_length,
        "completion_only_loss": True,
        "seed": args.seed,
        "train_loss": round(train_result.metrics.get("train_loss", 0), 4),
        "train_runtime_min": round(t_elapsed / 60, 1),
        "peak_vram_gb": round(peak_vram, 2),
        "total_steps": trainer.state.global_step,
        "best_eval_loss": round(trainer.state.best_metric, 4) if trainer.state.best_metric else None,
        "resumed_from": args.resume_from,
        "format": "prompt_completion",
        "eos_token": tokenizer.eos_token,
    }

    with open(os.path.join(args.output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output_dir, "log_history.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  Training Complete!")
    print(f"{'=' * 60}")
    print(f"  Train loss:      {summary['train_loss']}")
    print(f"  Best eval loss:  {summary['best_eval_loss']}")
    print(f"  Training time:   {summary['train_runtime_min']} min")
    print(f"  Peak VRAM:       {summary['peak_vram_gb']} GB")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
