#!/usr/bin/env python3
"""Wrapper for GSM8K analyzer SFT training and optional posterior recompute."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from gsm8k_learned_analyzer_utils import load_jsonl, validate_runtime_sft_example, write_json


DEFAULT_DATASET_DIR = "outputs/gsm8k_learned_analyzer/sft_data"
DEFAULT_OUTPUT_DIR = "outputs/gsm8k_learned_analyzer/sft_adapter"
DEFAULT_LOG_DIR = (
    "outputs/gsm8k_experiments/"
    "grpo_bayesian_prompted_qwen3b_train3000_n8_steps500_lambda10"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train analyzer SFT on GSM8K distillation data."
    )
    parser.add_argument("--dataset_dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--dataset_variant",
        choices=("runtime", "simple"),
        default="runtime",
        help="runtime keeps compatibility with the current learned reward module.",
    )
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--use_4bit", action="store_true")

    parser.add_argument("--run_recompute", action="store_true")
    parser.add_argument("--recompute_log_dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--recompute_output_dir", default="")
    parser.add_argument("--recompute_batch_size", type=int, default=8)
    parser.add_argument("--recompute_max_new_tokens", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    train_path = dataset_dir / f"{args.dataset_variant}_unified_train.jsonl"
    valid_path = dataset_dir / f"{args.dataset_variant}_unified_valid.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train split: {train_path}")
    if not valid_path.exists():
        raise FileNotFoundError(f"Missing valid split: {valid_path}")
    if args.dataset_variant == "runtime":
        for path in (train_path, valid_path):
            for row in load_jsonl(path):
                validate_runtime_sft_example(row)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_cmd = [
        "python3",
        "train_unified_analyzer_sft.py",
        "--model_name",
        args.model_name,
        "--train_path",
        str(train_path),
        "--val_path",
        str(valid_path),
        "--output_dir",
        str(output_dir),
        "--max_length",
        str(args.max_length),
        "--num_train_epochs",
        str(args.num_train_epochs),
        "--per_device_train_batch_size",
        str(args.per_device_train_batch_size),
        "--per_device_eval_batch_size",
        str(args.per_device_eval_batch_size),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--learning_rate",
        str(args.learning_rate),
        "--logging_steps",
        str(args.logging_steps),
        "--save_steps",
        str(args.save_steps),
        "--eval_steps",
        str(args.eval_steps),
        "--seed",
        str(args.seed),
    ]
    if args.bf16:
        train_cmd.append("--bf16")
    if args.use_4bit:
        train_cmd.append("--use_4bit")

    write_json(
        output_dir / "launcher_config.json",
        {
            "dataset_variant": args.dataset_variant,
            "train_path": str(train_path),
            "valid_path": str(valid_path),
            "train_cmd": train_cmd,
        },
    )

    print(json.dumps({"train_cmd": train_cmd}, ensure_ascii=False, indent=2))
    subprocess.run(train_cmd, check=True)

    if not args.run_recompute:
        return

    recompute_output_dir = (
        Path(args.recompute_output_dir)
        if args.recompute_output_dir
        else output_dir / "recompute_on_prompted_pool"
    )
    recompute_output_dir.mkdir(parents=True, exist_ok=True)
    debug_jsonl = Path(args.recompute_log_dir) / "bayesian_reward_debug.jsonl"
    recompute_cmd = [
        "python3",
        "recompute_posterior_with_learned_analyzer.py",
        "--input_debug_jsonl",
        str(debug_jsonl),
        "--output_dir",
        str(recompute_output_dir),
        "--model_name",
        args.model_name,
        "--adapter_path",
        str(output_dir),
        "--batch_size",
        str(args.recompute_batch_size),
        "--max_new_tokens",
        str(args.recompute_max_new_tokens),
    ]
    if args.bf16:
        recompute_cmd.append("--bf16")
    print(json.dumps({"recompute_cmd": recompute_cmd}, ensure_ascii=False, indent=2))
    subprocess.run(recompute_cmd, check=True)


if __name__ == "__main__":
    main()
