#!/usr/bin/env python3
"""Train a unified analyzer with DPO.

This trainer supports two experiment starts:

1. Base Qwen -> Analyzer DPO
   Omit --init_adapter_path and a fresh LoRA adapter is trained from the base model.

2. v0 SFT Analyzer -> Analyzer DPO
   Pass --init_adapter_path to continue from an existing SFT LoRA adapter.

The training data is expected to contain prompt-only chat messages plus
schema-perfect chosen/rejected JSON strings.

Important TRL compatibility note:
- Newer TRL versions expect beta/max_length/max_prompt_length/max_completion_length
  inside DPOConfig instead of direct DPOTrainer kwargs.
- Older TRL versions may accept some of these directly in DPOTrainer.
This script handles both patterns.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train unified analyzer DPO with LoRA."
    )
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--val_path", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument(
        "--init_adapter_path",
        default=None,
        help="Optional LoRA adapter to continue from. Use this for SFT -> DPO.",
    )
    parser.add_argument(
        "--reference_adapter_path",
        default=None,
        help=(
            "Optional frozen reference adapter. Defaults to --init_adapter_path when provided, "
            "otherwise the plain base model is used."
        ),
    )

    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--max_prompt_length", type=int, default=3584)
    parser.add_argument("--max_completion_length", type=int, default=512)

    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_rows", type=int, default=0)
    parser.add_argument("--max_val_rows", type=int, default=0)
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"Expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def render_chat_prompt(messages: list[dict[str, str]], tokenizer: Any) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def validate_pair_row(row: dict[str, Any]) -> tuple[bool, str]:
    if "chosen" not in row or "rejected" not in row:
        return False, "missing_preference_fields"

    chosen = str(row.get("chosen", "")).strip()
    rejected = str(row.get("rejected", "")).strip()
    if not chosen or not rejected:
        return False, "empty_preference"
    if chosen == rejected:
        return False, "identical_preference"

    for key in ("chosen", "rejected"):
        try:
            parsed = json.loads(str(row[key]))
        except json.JSONDecodeError:
            return False, f"{key}_invalid_json"
        if not isinstance(parsed, dict):
            return False, f"{key}_not_object"

    messages = row.get("prompt_messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return False, "missing_prompt_messages"

    for message in messages:
        if not isinstance(message, dict):
            return False, "prompt_message_not_dict"
        if message.get("role") not in {"system", "user", "assistant"}:
            return False, "invalid_role"
        if "content" not in message:
            return False, "missing_content"

    return True, "ok"


def preprocess_rows(
    rows: list[dict[str, Any]],
    *,
    tokenizer: Any,
    split_name: str,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    processed: list[dict[str, Any]] = []
    skip_reasons: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()

    for row in rows:
        ok, reason = validate_pair_row(row)
        if not ok:
            skip_reasons[reason] += 1
            continue

        prompt = render_chat_prompt(list(row["prompt_messages"]), tokenizer)
        processed_row = {
            "prompt": prompt,
            "chosen": str(row["chosen"]).strip(),
            "rejected": str(row["rejected"]).strip(),
            "task": str(row.get("task", "unknown")),
            "bucket_name": str(row.get("bucket_name", "unknown")),
            "pair_id": str(row.get("pair_id", "")),
        }
        processed.append(processed_row)
        task_counts[processed_row["task"]] += 1
        bucket_counts[processed_row["bucket_name"]] += 1

    print(
        f"[INFO] {split_name} rows={len(rows)} kept={len(processed)} skipped={len(rows) - len(processed)}"
    )
    if skip_reasons:
        print(f"[INFO] {split_name} skip reasons: {dict(skip_reasons)}")
    print(f"[INFO] {split_name} task distribution: {dict(sorted(task_counts.items()))}")
    print(f"[INFO] {split_name} bucket distribution: {dict(sorted(bucket_counts.items()))}")
    return processed, dict(sorted(task_counts.items())), dict(sorted(bucket_counts.items()))


def to_hf_dataset(rows: list[dict[str, Any]]):
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required. Install it with: pip install datasets") from exc
    return Dataset.from_list(rows)


def build_base_model_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
    }

    if args.use_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError(
                "bitsandbytes quantization requires transformers BitsAndBytesConfig."
            ) from exc

        compute_dtype = torch.bfloat16 if args.bf16 else torch.float16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        if torch.cuda.is_available():
            model_kwargs["device_map"] = {"": 0}
        return model_kwargs

    if args.bf16 and torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif args.fp16 and torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.float16
    else:
        model_kwargs["torch_dtype"] = "auto"

    if torch.cuda.is_available():
        model_kwargs["device_map"] = {"": 0}

    return model_kwargs


def import_peft():
    try:
        from peft import (
            LoraConfig,
            PeftModel,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
    except ImportError as exc:
        raise RuntimeError("peft is required. Install it with: pip install peft") from exc
    return LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training


def build_lora_config(args: argparse.Namespace):
    LoraConfig, _, _, _ = import_peft()
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


def mark_only_lora_trainable(model: Any) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "lora_" in name


def freeze_model(model: Any) -> Any:
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model


def load_trainable_model(args: argparse.Namespace):
    _, PeftModel, get_peft_model, prepare_model_for_kbit_training = import_peft()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **build_base_model_kwargs(args),
    )

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    if args.init_adapter_path:
        print(f"[INFO] loading trainable init adapter: {args.init_adapter_path}")
        try:
            model = PeftModel.from_pretrained(
                model,
                args.init_adapter_path,
                is_trainable=True,
            )
        except TypeError:
            model = PeftModel.from_pretrained(model, args.init_adapter_path)
            mark_only_lora_trainable(model)
    else:
        print("[INFO] no init_adapter_path provided. Training fresh LoRA adapter from base model.")
        model = get_peft_model(model, build_lora_config(args))

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    model.print_trainable_parameters()
    return model


def load_reference_model(args: argparse.Namespace):
    _, PeftModel, _, _ = import_peft()

    reference_adapter_path = args.reference_adapter_path or args.init_adapter_path

    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **build_base_model_kwargs(args),
    )

    if reference_adapter_path:
        print(f"[INFO] loading frozen reference adapter: {reference_adapter_path}")
        ref_model = PeftModel.from_pretrained(ref_model, reference_adapter_path)
    else:
        print("[INFO] no reference adapter provided. Using frozen base model as reference.")

    return freeze_model(ref_model)


def get_eval_strategy_kwarg(has_eval_dataset: bool) -> dict[str, str]:
    """Return the correct eval strategy kwarg for installed Transformers version."""
    strategy_value = "steps" if has_eval_dataset else "no"
    signature = inspect.signature(TrainingArguments.__init__)

    if "eval_strategy" in signature.parameters:
        return {"eval_strategy": strategy_value}
    return {"evaluation_strategy": strategy_value}


def try_import_dpo_config():
    try:
        from trl import DPOConfig
        return DPOConfig
    except Exception:
        return None


def filter_kwargs_for_class_init(cls: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep kwargs accepted by cls.__init__.

    If cls.__init__ has **kwargs, keep everything.
    """
    signature = inspect.signature(cls.__init__)
    params = signature.parameters

    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return kwargs

    return {key: value for key, value in kwargs.items() if key in params}


def build_training_arguments(
    args: argparse.Namespace,
    *,
    has_eval_dataset: bool,
):
    """Build DPOConfig when available; otherwise fall back to TrainingArguments.

    Newer TRL expects DPO-specific fields such as beta/max_length inside DPOConfig,
    not as direct DPOTrainer kwargs.
    """
    common_kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_total_limit": args.save_total_limit,
        "bf16": bool(args.bf16 and torch.cuda.is_available()),
        "fp16": bool(args.fp16 and torch.cuda.is_available() and not args.bf16),
        "report_to": "none",
        "remove_unused_columns": False,
        "save_strategy": "steps",
        "logging_strategy": "steps",
        "gradient_checkpointing": args.gradient_checkpointing,
        "seed": args.seed,
    }
    common_kwargs.update(get_eval_strategy_kwarg(has_eval_dataset))

    DPOConfig = try_import_dpo_config()
    if DPOConfig is not None:
        dpo_kwargs = dict(common_kwargs)
        dpo_kwargs.update(
            {
                "beta": args.beta,
                "max_length": args.max_length,
                "max_prompt_length": args.max_prompt_length,
                "max_completion_length": args.max_completion_length,
            }
        )
        dpo_kwargs = filter_kwargs_for_class_init(DPOConfig, dpo_kwargs)
        print("[INFO] using trl.DPOConfig for training args")
        return DPOConfig(**dpo_kwargs)

    print("[WARN] trl.DPOConfig not available. Falling back to transformers.TrainingArguments.")
    training_kwargs = filter_kwargs_for_class_init(TrainingArguments, common_kwargs)
    return TrainingArguments(**training_kwargs)


def build_dpo_trainer(
    model: Any,
    ref_model: Any,
    *,
    tokenizer: Any,
    training_args: Any,
    train_dataset: Any,
    eval_dataset: Any,
    args: argparse.Namespace,
):
    try:
        from trl import DPOTrainer
    except ImportError as exc:
        raise RuntimeError("trl is required. Install it with: pip install trl") from exc

    signature = inspect.signature(DPOTrainer.__init__)
    params = signature.parameters

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "ref_model": ref_model,
        "args": training_args,
        "train_dataset": train_dataset,
    }

    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset

    if "tokenizer" in params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in params:
        trainer_kwargs["processing_class"] = tokenizer

    # Older TRL versions accepted beta/max lengths directly in DPOTrainer.
    # Newer TRL versions expect these in DPOConfig, so only pass them if accepted.
    if "beta" in params:
        trainer_kwargs["beta"] = args.beta

    if "max_length" in params:
        trainer_kwargs["max_length"] = args.max_length
    if "max_prompt_length" in params:
        trainer_kwargs["max_prompt_length"] = args.max_prompt_length
    if "max_completion_length" in params:
        trainer_kwargs["max_completion_length"] = args.max_completion_length
    elif "max_target_length" in params:
        trainer_kwargs["max_target_length"] = args.max_completion_length

    if "generate_during_eval" in params:
        trainer_kwargs["generate_during_eval"] = False

    # Drop any unsupported kwargs just in case.
    if not any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        trainer_kwargs = {
            key: value
            for key, value in trainer_kwargs.items()
            if key in params
        }

    print("[INFO] DPOTrainer kwargs:", sorted(trainer_kwargs.keys()))
    return DPOTrainer(**trainer_kwargs)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    train_rows = load_jsonl(args.train_path)
    val_rows = load_jsonl(args.val_path)

    if args.max_train_rows and args.max_train_rows > 0:
        train_rows = train_rows[: args.max_train_rows]
    if args.max_val_rows and args.max_val_rows > 0:
        val_rows = val_rows[: args.max_val_rows]

    print(f"[INFO] loaded train rows={len(train_rows)} from {args.train_path}")
    print(f"[INFO] loaded val rows={len(val_rows)} from {args.val_path}")

    tokenizer = load_tokenizer(args.model_name)

    processed_train, train_task_counts, train_bucket_counts = preprocess_rows(
        train_rows,
        tokenizer=tokenizer,
        split_name="train",
    )
    processed_val, val_task_counts, val_bucket_counts = preprocess_rows(
        val_rows,
        tokenizer=tokenizer,
        split_name="val",
    )

    if not processed_train:
        raise RuntimeError("Train dataset is empty after preprocessing.")
    if not processed_val:
        print("[WARN] Val dataset is empty after preprocessing. Evaluation may be skipped.")

    train_dataset = to_hf_dataset(processed_train)
    eval_dataset = to_hf_dataset(processed_val) if processed_val else None

    model = load_trainable_model(args)
    ref_model = load_reference_model(args)

    training_args = build_training_arguments(
        args,
        has_eval_dataset=eval_dataset is not None,
    )

    trainer = build_dpo_trainer(
        model,
        ref_model,
        tokenizer=tokenizer,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=args,
    )

    trainer.train()

    print(f"[INFO] saving LoRA adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    summary = {
        "model_name": args.model_name,
        "train_path": args.train_path,
        "val_path": args.val_path,
        "output_dir": args.output_dir,
        "init_adapter_path": args.init_adapter_path,
        "reference_adapter_path": args.reference_adapter_path or args.init_adapter_path,
        "beta": args.beta,
        "max_length": args.max_length,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "use_4bit": args.use_4bit,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "train_examples_after_filter": len(processed_train),
        "val_examples_after_filter": len(processed_val),
        "train_task_distribution": train_task_counts,
        "val_task_distribution": val_task_counts,
        "train_bucket_distribution": train_bucket_counts,
        "val_bucket_distribution": val_bucket_counts,
    }

    summary_path = Path(args.output_dir) / "train_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"[INFO] wrote train summary to {summary_path}")


if __name__ == "__main__":
    main()