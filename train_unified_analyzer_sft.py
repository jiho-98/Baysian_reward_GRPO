#!/usr/bin/env python3
"""Train a unified multi-task analyzer with LoRA SFT.

Expected input JSONL format:
Each row should contain:
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "{...json...}"}
  ],
  "task": "evidence_judge" or "prior_judge"
}

This script trains only on the assistant response tokens.
The prompt tokens are masked with label = -100.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except ImportError as exc:
    raise RuntimeError(
        "peft is required. Install it with: pip install peft"
    ) from exc


IGNORE_INDEX = -100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train unified analyzer SFT with LoRA."
    )

    parser.add_argument("--model_name", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--val_path", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)

    parser.add_argument("--learning_rate", type=float, default=2e-4)
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


def render_prompt(messages: list[dict[str, str]], tokenizer: Any) -> str:
    """Render all non-assistant-target messages as a chat prompt."""
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


def normalize_token_sequence(token_ids: Any) -> list[int]:
    """Robustly convert tokenizer outputs / tensors / lists into list[int]."""

    if token_ids is None:
        return []

    # HuggingFace BatchEncoding or normal dict
    if isinstance(token_ids, dict):
        token_ids = token_ids["input_ids"]

    # Some BatchEncoding-like objects expose .data
    if hasattr(token_ids, "data") and isinstance(getattr(token_ids, "data"), dict):
        data = getattr(token_ids, "data")
        if "input_ids" in data:
            token_ids = data["input_ids"]

    # torch tensor / numpy array
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()

    # Flatten [[...]] into [...]
    if (
        isinstance(token_ids, list)
        and len(token_ids) == 1
        and isinstance(token_ids[0], list)
    ):
        token_ids = token_ids[0]

    normalized: list[int] = []
    for item in token_ids:
        if hasattr(item, "item"):
            item = item.item()
        normalized.append(int(item))

    return normalized


def tokenize_text(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
    )
    return normalize_token_sequence(encoded["input_ids"])


def validate_messages(row: dict[str, Any]) -> tuple[bool, str]:
    messages = row.get("messages")

    if not isinstance(messages, list):
        return False, "missing_messages"

    if len(messages) < 2:
        return False, "too_few_messages"

    for message in messages:
        if not isinstance(message, dict):
            return False, "message_not_dict"
        if message.get("role") not in {"system", "user", "assistant"}:
            return False, "invalid_role"
        if "content" not in message:
            return False, "missing_content"

    if messages[-1].get("role") != "assistant":
        return False, "last_message_not_assistant"

    assistant_content = str(messages[-1].get("content", "")).strip()
    if not assistant_content:
        return False, "empty_assistant"

    user_exists = any(message.get("role") == "user" for message in messages[:-1])
    if not user_exists:
        return False, "missing_user"

    return True, "ok"


class SupervisedDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        max_length: int,
        split_name: str,
    ) -> None:
        self.examples: list[dict[str, Any]] = []
        self.skip_reasons: Counter[str] = Counter()

        for row in rows:
            ok, reason = validate_messages(row)
            if not ok:
                self.skip_reasons[reason] += 1
                continue

            item = self.build_example(row, tokenizer, max_length)
            if item is None:
                continue

            self.examples.append(item)

        print(
            f"[INFO] {split_name} rows={len(rows)} "
            f"kept={len(self.examples)} skipped={len(rows) - len(self.examples)}"
        )

        if self.skip_reasons:
            print(f"[INFO] {split_name} skip reasons: {dict(self.skip_reasons)}")

    def build_example(
        self,
        row: dict[str, Any],
        tokenizer: Any,
        max_length: int,
    ) -> Optional[dict[str, torch.Tensor]]:

        messages = row["messages"]
        prompt_messages = messages[:-1]
        assistant_text = str(messages[-1]["content"]).strip()

        # Ensure EOS after target JSON.
        if tokenizer.eos_token:
            target_text = assistant_text + tokenizer.eos_token
        else:
            target_text = assistant_text

        try:
            prompt_text = render_prompt(prompt_messages, tokenizer)
        except Exception:
            self.skip_reasons["chat_template_error"] += 1
            return None

        try:
            prompt_ids = tokenize_text(tokenizer, prompt_text)
            target_ids = tokenize_text(tokenizer, target_text)
        except Exception:
            self.skip_reasons["tokenization_error"] += 1
            return None

        if not prompt_ids:
            self.skip_reasons["empty_prompt_ids"] += 1
            return None

        if not target_ids:
            self.skip_reasons["empty_target_ids"] += 1
            return None

        # If the target alone is too long, truncate target.
        # This should be rare because targets are JSON labels.
        if len(target_ids) >= max_length:
            target_ids = target_ids[: max_length - 1]
            if tokenizer.eos_token_id is not None:
                target_ids.append(int(tokenizer.eos_token_id))

        # Keep the full target whenever possible.
        # If too long, truncate from the left side of the prompt.
        max_prompt_len = max_length - len(target_ids)
        if max_prompt_len <= 0:
            self.skip_reasons["target_too_long"] += 1
            return None

        if len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[-max_prompt_len:]

        input_ids = prompt_ids + target_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + target_ids
        attention_mask = [1] * len(input_ids)

        if len(input_ids) == 0:
            self.skip_reasons["empty_input"] += 1
            return None

        if len(input_ids) > max_length:
            self.skip_reasons["still_too_long"] += 1
            return None

        # If all labels are ignored, training is impossible.
        if all(label == IGNORE_INDEX for label in labels):
            self.skip_reasons["all_labels_ignored"] += 1
            return None

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]


class DataCollatorForCausalSFT:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id

    def __call__(
        self,
        features: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:

        max_len = max(feature["input_ids"].shape[0] for feature in features)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for feature in features:
            input_ids = feature["input_ids"]
            attention_mask = feature["attention_mask"]
            labels = feature["labels"]

            pad_len = max_len - input_ids.shape[0]

            if pad_len > 0:
                input_ids = torch.cat(
                    [
                        input_ids,
                        torch.full(
                            (pad_len,),
                            int(self.pad_token_id),
                            dtype=torch.long,
                        ),
                    ],
                    dim=0,
                )
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.zeros((pad_len,), dtype=torch.long),
                    ],
                    dim=0,
                )
                labels = torch.cat(
                    [
                        labels,
                        torch.full((pad_len,), IGNORE_INDEX, dtype=torch.long),
                    ],
                    dim=0,
                )

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)

        return {
            "input_ids": torch.stack(batch_input_ids, dim=0),
            "attention_mask": torch.stack(batch_attention_mask, dim=0),
            "labels": torch.stack(batch_labels, dim=0),
        }


def load_model(args: argparse.Namespace):
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
        model_kwargs["device_map"] = {"": 0}
    else:
        if args.bf16 and torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif args.fp16 and torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = "auto"

        if torch.cuda.is_available():
            model_kwargs["device_map"] = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_kwargs,
    )

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
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

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def build_training_arguments(args: argparse.Namespace) -> TrainingArguments:
    kwargs: dict[str, Any] = {
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
        "seed": args.seed,
    }

    # transformers versions differ:
    # New versions use eval_strategy; older versions use evaluation_strategy.
    try:
        return TrainingArguments(
            **kwargs,
            eval_strategy="steps",
        )
    except TypeError:
        return TrainingArguments(
            **kwargs,
            evaluation_strategy="steps",
        )


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

    train_dataset = SupervisedDataset(
        train_rows,
        tokenizer=tokenizer,
        max_length=args.max_length,
        split_name="train",
    )
    val_dataset = SupervisedDataset(
        val_rows,
        tokenizer=tokenizer,
        max_length=args.max_length,
        split_name="val",
    )

    if len(train_dataset) == 0:
        raise RuntimeError(
            "Train dataset is empty after preprocessing. "
            "Check messages format or increase --max_length."
        )

    if len(val_dataset) == 0:
        print("[WARN] Val dataset is empty after preprocessing. Evaluation may fail.")

    model = load_model(args)
    data_collator = DataCollatorForCausalSFT(tokenizer)

    training_args = build_training_arguments(args)

    trainer_kwargs = {
    "model": model,
    "args": training_args,
    "train_dataset": train_dataset,
    "eval_dataset": val_dataset if len(val_dataset) > 0 else None,
    "data_collator": data_collator,
    }

    try:
        trainer = Trainer(
            **trainer_kwargs,
            processing_class=tokenizer,
        )
    except TypeError:
        trainer = Trainer(
            **trainer_kwargs,
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
        "max_length": args.max_length,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "use_4bit": args.use_4bit,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "train_examples_after_filter": len(train_dataset),
        "val_examples_after_filter": len(val_dataset),
    }

    summary_path = Path(args.output_dir) / "train_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"[INFO] wrote train summary to {summary_path}")


if __name__ == "__main__":
    main()