#!/usr/bin/env python3
"""Controlled LoRA port of VeriFree for this repository's math RL setup.

This script is intentionally not a plain "reward function plugged into GRPO".
It ports the core VeriFree training objective into the same data, prompt, LoRA,
and budget surface used by the local BPR/answer-only experiments:

1. Sample multiple reasoning prefixes for each problem.
2. Append the gold answer suffix and compute log P(gold answer | prompt, prefix).
3. Use that reference-answer likelihood as the group-relative reward.
4. Update the sampled reasoning tokens with a policy-gradient loss.
5. Update the appended gold-answer tokens with a reward-weighted SFT loss.

The implementation follows the public VeriFree idea while avoiding the OAT
full-finetuning stack, so Qwen3 LoRA runs can be compared under the same local
budget as BPR.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from Answer_only_GRPO import (
    DEFAULT_DATASET_NAME,
    DEFAULT_EVAL_METADATA_PATH,
    DEFAULT_MODEL_NAME,
    DEFAULT_TRAIN_METADATA_PATH,
    add_bool_arg,
    build_peft_config,
    default_bf16_enabled,
    ensure_output_dir,
    load_tokenizer,
    load_training_data,
    set_seed,
    write_json,
)


DEFAULT_OUTPUT_DIR = "outputs/verifree_lora_qwen3b_bigmath"
DEFAULT_DEBUG_JSONL = "verifree_reward_debug.jsonl"
KST = timezone(timedelta(hours=9))


@dataclass
class VeriFreeSample:
    problem: str
    gold_answer: str
    prompt_text: str
    generated_text: str
    think_text: str
    answer_text: str
    prompt_ids: list[int]
    think_ids: list[int]
    answer_ids: list[int]
    group_id: int
    rollout_id: int
    answer_logp_sum: float = float("-inf")
    answer_logp_mean: float = float("-inf")
    reward: float = 0.0
    advantage: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VeriFree controlled LoRA training on this repo's fixed metadata."
    )
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--min_solve_rate", type=float, default=0.0)
    parser.add_argument("--max_solve_rate", type=float, default=1.0)
    parser.add_argument("--train_size", type=int, default=None)
    parser.add_argument("--eval_size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Number of prompts sampled per policy update.",
    )
    parser.add_argument(
        "--mini_train_batch_size",
        type=int,
        default=4,
        help="Number of rollout samples per forward/backward microbatch.",
    )
    parser.add_argument(
        "--generation_prompt_batch_size",
        type=int,
        default=0,
        help="Split rollout generation into prompt chunks. 0 means use the full prompt batch.",
    )
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_completion_length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)

    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--progress_interval_percent",
        type=int,
        default=10,
        help="Print progress every N percent of max_steps.",
    )

    parser.add_argument(
        "--advantage_type",
        choices=["rloo", "grpo"],
        default="rloo",
        help="VeriFree official run uses rloo; grpo uses group mean subtraction.",
    )
    parser.add_argument(
        "--reward_source",
        choices=["logp", "p", "mean_logp", "p_mean"],
        default="p",
        help="VeriFree official run uses p=exp(sum logp). p_mean is a stable ablation.",
    )
    parser.add_argument(
        "--reward_scale",
        type=float,
        default=1.0,
        help="Scale log-probability before converting to reward, matching VeriFree/OAT reward_scale.",
    )
    parser.add_argument(
        "--sft_coef_source",
        choices=["reward", "adv", "1"],
        default="reward",
        help="VeriFree official run uses reward.",
    )
    add_bool_arg(
        parser,
        "normalize_advantages",
        False,
        "Normalize group-relative advantages by group std.",
    )
    add_bool_arg(
        parser,
        "length_normalize_pg",
        False,
        "Use mean reasoning logprob instead of summed reasoning logprob in PG loss.",
    )
    parser.add_argument(
        "--answer_suffix_template",
        default="\n\n[Final Answer]\n\\boxed{{{answer}}}",
        help="Gold answer suffix appended after the sampled reasoning prefix.",
    )
    parser.add_argument(
        "--max_answer_tokens",
        type=int,
        default=128,
        help="Drop/update rows whose rendered gold answer suffix exceeds this many tokens.",
    )

    add_bool_arg(parser, "use_lora", True, "Enable LoRA adapters.")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    add_bool_arg(parser, "bf16", True, "Use bf16 when supported.")
    add_bool_arg(parser, "gradient_checkpointing", True, "Enable gradient checkpointing.")
    add_bool_arg(
        parser,
        "use_fixed_metadata",
        True,
        "Load fixed selected_train/selected_eval metadata for a fair comparison.",
    )
    parser.add_argument("--train_metadata_path", default=DEFAULT_TRAIN_METADATA_PATH)
    parser.add_argument("--eval_metadata_path", default=DEFAULT_EVAL_METADATA_PATH)
    parser.add_argument("--debug_jsonl", default=None)
    add_bool_arg(parser, "smoke_test_only", False, "Run one train step and exit.")
    parser.add_argument(
        "--smoke_test_examples",
        type=int,
        default=8,
        help="Compatibility field used by the shared metadata loader.",
    )

    return parser.parse_args()


def append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def format_kst_timestamp(seconds_from_now: float) -> str:
    finish_at = datetime.now(KST) + timedelta(seconds=max(0.0, float(seconds_from_now)))
    return finish_at.strftime("%Y-%m-%d %H:%M:%S KST")


def batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def split_reasoning_prefix(generated_text: str) -> str:
    """Keep generated reasoning, but replace any generated final answer with gold."""
    text = str(generated_text or "")
    match = re.search(r"\n?\s*\[Final Answer\]\s*", text, flags=re.IGNORECASE)
    if match:
        prefix = text[: match.start()]
    else:
        prefix = text
    return prefix.rstrip()


def render_answer_suffix(template: str, answer: Any) -> str:
    answer_text = str(answer if answer is not None else "").strip()
    return template.format(answer=answer_text)


def encode_no_special(tokenizer: Any, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def collate_token_samples(
    samples: list[VeriFreeSample],
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    full_ids: list[list[int]] = []
    think_masks: list[list[float]] = []
    answer_masks: list[list[float]] = []

    max_length = 0
    for sample in samples:
        ids = sample.prompt_ids + sample.think_ids + sample.answer_ids
        if len(ids) < 2:
            ids = ids + [pad_token_id]
        full_ids.append(ids)
        max_length = max(max_length, len(ids))

        logp_positions = max(len(ids) - 1, 1)
        think_mask = [0.0] * logp_positions
        answer_mask = [0.0] * logp_positions
        prompt_len = len(sample.prompt_ids)
        think_len = len(sample.think_ids)
        answer_len = len(sample.answer_ids)

        think_start = max(prompt_len - 1, 0)
        think_end = max(prompt_len + think_len - 1, think_start)
        for pos in range(think_start, min(think_end, logp_positions)):
            think_mask[pos] = 1.0

        answer_start = max(prompt_len + think_len - 1, 0)
        answer_end = max(prompt_len + think_len + answer_len - 1, answer_start)
        for pos in range(answer_start, min(answer_end, logp_positions)):
            answer_mask[pos] = 1.0

        think_masks.append(think_mask)
        answer_masks.append(answer_mask)

    input_rows: list[list[int]] = []
    attention_rows: list[list[int]] = []
    think_rows: list[list[float]] = []
    answer_rows: list[list[float]] = []
    for ids, think_mask, answer_mask in zip(full_ids, think_masks, answer_masks):
        pad_len = max_length - len(ids)
        input_rows.append(ids + [pad_token_id] * pad_len)
        attention_rows.append([1] * len(ids) + [0] * pad_len)
        logp_pad_len = max(max_length - 1 - len(think_mask), 0)
        think_rows.append(think_mask + [0.0] * logp_pad_len)
        answer_rows.append(answer_mask + [0.0] * logp_pad_len)

    input_ids = torch.tensor(input_rows, dtype=torch.long, device=device)
    attention_mask = torch.tensor(attention_rows, dtype=torch.long, device=device)
    think_mask_t = torch.tensor(think_rows, dtype=torch.float32, device=device)
    answer_mask_t = torch.tensor(answer_rows, dtype=torch.float32, device=device)
    return input_ids, attention_mask, think_mask_t, answer_mask_t


def sequence_logprobs(model: Any, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :].float()
    target_ids = input_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)


def reward_from_logp(sum_logp: float, mean_logp: float, reward_source: str, reward_scale: float) -> float:
    scaled_sum_logp = float(sum_logp) * float(reward_scale)
    scaled_mean_logp = float(mean_logp) * float(reward_scale)
    if reward_source == "logp":
        return scaled_sum_logp
    if reward_source == "mean_logp":
        return scaled_mean_logp
    if reward_source == "p":
        return float(math.exp(max(min(scaled_sum_logp, 0.0), -80.0)))
    if reward_source == "p_mean":
        return float(math.exp(max(min(scaled_mean_logp, 0.0), -80.0)))
    raise RuntimeError(f"Unsupported reward_source={reward_source}")


def assign_group_advantages(samples: list[VeriFreeSample], args: argparse.Namespace) -> None:
    group_size = int(args.num_generations)
    for group in batched(samples, group_size):
        if not group:
            continue
        rewards = torch.tensor([sample.reward for sample in group], dtype=torch.float32)
        if args.advantage_type == "rloo" and len(group) > 1:
            # Same algebra as the official VeriFree implementation:
            # (r_i - group_mean) * n / (n - 1), i.e. leave-one-out baseline.
            advantages = (rewards - rewards.mean()) * (len(group) / (len(group) - 1))
        else:
            advantages = rewards - rewards.mean()
        if args.normalize_advantages and len(group) > 1:
            advantages = advantages / rewards.std(unbiased=False).clamp(min=1e-8)
        for sample, advantage in zip(group, advantages.tolist()):
            sample.advantage = float(advantage)


def load_model(args: argparse.Namespace, device: torch.device):
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=dtype if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    if args.use_lora:
        model = get_peft_model(model, build_peft_config(args))
        model.print_trainable_parameters()
    return model


def generate_rollout_chunk(
    *,
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
) -> list[VeriFreeSample]:
    model.eval()
    prompts = [row["prompt"] for row in rows]
    answers = [str(row["answer"]) for row in rows]
    problems = [str(row.get("problem", "")) for row in rows]
    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=args.max_prompt_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        output_ids = model.generate(
            **encoded,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_completion_length,
            num_return_sequences=args.num_generations,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    samples: list[VeriFreeSample] = []
    prompt_token_ids = encoded["input_ids"].detach().cpu().tolist()
    attention_rows = encoded["attention_mask"].detach().cpu().tolist()
    # With left padding, generate() returns the full padded input prefix. The
    # completion begins after the padded input width, not after attention.sum().
    generation_prefix_len = int(encoded["input_ids"].shape[1])
    expanded_prompt_lens = [generation_prefix_len] * (len(rows) * args.num_generations)
    expanded_prompts = []
    for prompt in prompts:
        expanded_prompts.extend([prompt] * args.num_generations)

    for flat_index in range(output_ids.shape[0]):
        group_id = flat_index // args.num_generations
        rollout_id = flat_index % args.num_generations
        prompt_text = expanded_prompts[flat_index]
        prompt_len = expanded_prompt_lens[flat_index]
        completion_ids = output_ids[flat_index, prompt_len:].detach().cpu().tolist()
        generated_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        think_text = split_reasoning_prefix(generated_text)
        answer_text = render_answer_suffix(args.answer_suffix_template, answers[group_id])

        prompt_ids = prompt_token_ids[group_id]
        prompt_ids = [token_id for token_id, mask in zip(prompt_ids, attention_rows[group_id]) if mask]
        prompt_ids = prompt_ids[-args.max_prompt_length :]
        think_ids = encode_no_special(tokenizer, think_text)
        think_ids = think_ids[: args.max_completion_length]
        answer_ids = encode_no_special(tokenizer, answer_text)
        if len(answer_ids) > args.max_answer_tokens:
            answer_ids = answer_ids[: args.max_answer_tokens]

        samples.append(
            VeriFreeSample(
                problem=problems[group_id],
                gold_answer=answers[group_id],
                prompt_text=prompt_text,
                generated_text=generated_text,
                think_text=think_text,
                answer_text=answer_text,
                prompt_ids=prompt_ids,
                think_ids=think_ids,
                answer_ids=answer_ids,
                group_id=group_id,
                rollout_id=rollout_id,
            )
        )
    return samples


def generate_rollouts(
    *,
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
) -> list[VeriFreeSample]:
    chunk_size = int(args.generation_prompt_batch_size or len(rows))
    if chunk_size <= 0:
        chunk_size = len(rows)
    if chunk_size >= len(rows):
        return generate_rollout_chunk(
            model=model,
            tokenizer=tokenizer,
            rows=rows,
            args=args,
            device=device,
        )

    all_samples: list[VeriFreeSample] = []
    for row_chunk in batched(rows, chunk_size):
        offset = len(all_samples) // int(args.num_generations)
        chunk_samples = generate_rollout_chunk(
            model=model,
            tokenizer=tokenizer,
            rows=row_chunk,
            args=args,
            device=device,
        )
        for sample in chunk_samples:
            sample.group_id += offset
        all_samples.extend(chunk_samples)
    return all_samples


def compute_rewards(
    *,
    model: Any,
    samples: list[VeriFreeSample],
    args: argparse.Namespace,
    pad_token_id: int,
    device: torch.device,
) -> None:
    model.eval()
    with torch.inference_mode():
        for micro_samples in batched(samples, args.mini_train_batch_size):
            input_ids, attention_mask, _, answer_mask = collate_token_samples(
                micro_samples,
                pad_token_id,
                device,
            )
            logps = sequence_logprobs(model, input_ids, attention_mask)
            answer_logp_sum = (logps * answer_mask).sum(dim=1)
            answer_token_count = answer_mask.sum(dim=1).clamp(min=1.0)
            answer_logp_mean = answer_logp_sum / answer_token_count
            for sample, sum_logp, mean_logp in zip(
                micro_samples,
                answer_logp_sum.detach().cpu().tolist(),
                answer_logp_mean.detach().cpu().tolist(),
            ):
                sample.answer_logp_sum = float(sum_logp)
                sample.answer_logp_mean = float(mean_logp)
                if not sample.answer_ids:
                    sample.reward = 0.0
                else:
                    sample.reward = reward_from_logp(
                        sample.answer_logp_sum,
                        sample.answer_logp_mean,
                        args.reward_source,
                        args.reward_scale,
                    )
    assign_group_advantages(samples, args)


def train_on_samples(
    *,
    model: Any,
    optimizer: torch.optim.Optimizer,
    samples: list[VeriFreeSample],
    args: argparse.Namespace,
    pad_token_id: int,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_samples = max(len(samples), 1)
    pg_values: list[float] = []
    sft_values: list[float] = []
    loss_values: list[float] = []
    think_lengths: list[int] = []
    answer_lengths: list[int] = []

    for micro_samples in batched(samples, args.mini_train_batch_size):
        input_ids, attention_mask, think_mask, answer_mask = collate_token_samples(
            micro_samples,
            pad_token_id,
            device,
        )
        logps = sequence_logprobs(model, input_ids, attention_mask)

        advantages = torch.tensor(
            [sample.advantage for sample in micro_samples],
            dtype=torch.float32,
            device=device,
        )
        rewards = torch.tensor(
            [sample.reward for sample in micro_samples],
            dtype=torch.float32,
            device=device,
        )

        think_logp_sum = (logps * think_mask).sum(dim=1)
        if args.length_normalize_pg:
            think_logp_sum = think_logp_sum / think_mask.sum(dim=1).clamp(min=1.0)
        answer_logp_sum = (logps * answer_mask).sum(dim=1)

        pg_loss_per_sample = -advantages.detach() * think_logp_sum
        if args.sft_coef_source == "reward":
            sft_weights = rewards.detach()
        elif args.sft_coef_source == "adv":
            sft_weights = advantages.detach()
        else:
            sft_weights = torch.ones_like(rewards)
        sft_loss_per_sample = -sft_weights * answer_logp_sum
        loss = (pg_loss_per_sample + sft_loss_per_sample).sum() / total_samples
        loss.backward()

        pg_values.extend(pg_loss_per_sample.detach().cpu().tolist())
        sft_values.extend(sft_loss_per_sample.detach().cpu().tolist())
        loss_values.extend((pg_loss_per_sample + sft_loss_per_sample).detach().cpu().tolist())
        think_lengths.extend(int(sample.think_ids.__len__()) for sample in micro_samples)
        answer_lengths.extend(int(sample.answer_ids.__len__()) for sample in micro_samples)

    if args.max_grad_norm and args.max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    return {
        "loss": float(sum(loss_values) / max(len(loss_values), 1)),
        "pg_loss": float(sum(pg_values) / max(len(pg_values), 1)),
        "sft_loss": float(sum(sft_values) / max(len(sft_values), 1)),
        "mean_reward": float(sum(sample.reward for sample in samples) / total_samples),
        "mean_advantage": float(sum(sample.advantage for sample in samples) / total_samples),
        "mean_answer_logp_sum": float(sum(sample.answer_logp_sum for sample in samples) / total_samples),
        "mean_answer_logp_mean": float(sum(sample.answer_logp_mean for sample in samples) / total_samples),
        "mean_think_tokens": float(sum(think_lengths) / max(len(think_lengths), 1)),
        "mean_answer_tokens": float(sum(answer_lengths) / max(len(answer_lengths), 1)),
    }


def make_row_batch(train_rows: list[dict[str, Any]], order: list[int], cursor: int, batch_size: int) -> tuple[list[dict[str, Any]], int]:
    if not train_rows:
        raise RuntimeError("No training rows available.")
    batch: list[dict[str, Any]] = []
    while len(batch) < batch_size:
        if cursor >= len(order):
            random.shuffle(order)
            cursor = 0
        batch.append(train_rows[order[cursor]])
        cursor += 1
    return batch, cursor


def save_checkpoint(model: Any, tokenizer: Any, output_dir: Path, step: int) -> None:
    checkpoint_dir = output_dir / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)


def main() -> None:
    args = parse_args()
    args.bf16 = bool(args.bf16 and default_bf16_enabled())
    if args.smoke_test_only:
        args.max_steps = min(args.max_steps, 1)
    if args.num_generations <= 1:
        raise SystemExit("--num_generations must be greater than 1.")
    if args.per_device_train_batch_size <= 0:
        raise SystemExit("--per_device_train_batch_size must be positive.")
    if args.mini_train_batch_size <= 0:
        raise SystemExit("--mini_train_batch_size must be positive.")

    output_dir = ensure_output_dir(args.output_dir)
    debug_path = Path(args.debug_jsonl) if args.debug_jsonl else output_dir / DEFAULT_DEBUG_JSONL
    if debug_path.exists():
        debug_path.unlink()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(args.model_name)
    train_dataset, _, _, dataset_stats, metadata_context = load_training_data(args, tokenizer, output_dir)
    train_rows = [dict(train_dataset[index]) for index in range(len(train_dataset))]

    model = load_model(args, device)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    config_payload = {
        **vars(args),
        "method": "VeriFree controlled LoRA port",
        "official_reference": "sail-sg/VeriFree",
        "objective": "PG on sampled reasoning prefix + reward-weighted SFT on appended gold answer",
        "dataset_stats": dataset_stats,
        "use_fixed_metadata": metadata_context["used_fixed_metadata"],
        "use_fixed_metadata_requested": args.use_fixed_metadata,
        "fixed_train_metadata_path": metadata_context["fixed_train_metadata_path"],
        "fixed_eval_metadata_path": metadata_context["fixed_eval_metadata_path"],
        "fixed_train_metadata_sha256": metadata_context["fixed_train_metadata_sha256"],
        "fixed_eval_metadata_sha256": metadata_context["fixed_eval_metadata_sha256"],
        "device": str(device),
        "debug_jsonl": str(debug_path),
    }
    write_json(output_dir / "training_config.json", config_payload)
    print("[INFO] final resolved configuration:")
    print(json.dumps(config_payload, ensure_ascii=False, indent=2, sort_keys=True))

    order = list(range(len(train_rows)))
    random.shuffle(order)
    cursor = 0
    start_time = time.time()
    progress_every = max(1, int(args.max_steps * max(args.progress_interval_percent, 1) / 100))
    if args.max_steps < progress_every:
        progress_every = args.max_steps

    for step in range(1, args.max_steps + 1):
        step_start = time.time()
        row_batch, cursor = make_row_batch(
            train_rows,
            order,
            cursor,
            args.per_device_train_batch_size,
        )
        samples = generate_rollouts(
            model=model,
            tokenizer=tokenizer,
            rows=row_batch,
            args=args,
            device=device,
        )
        compute_rewards(
            model=model,
            samples=samples,
            args=args,
            pad_token_id=tokenizer.pad_token_id,
            device=device,
        )
        metrics = train_on_samples(
            model=model,
            optimizer=optimizer,
            samples=samples,
            args=args,
            pad_token_id=tokenizer.pad_token_id,
            device=device,
        )
        step_time = time.time() - step_start

        for sample in samples:
            append_jsonl_row(
                debug_path,
                {
                    "step": step,
                    "group_id": sample.group_id,
                    "rollout_id": sample.rollout_id,
                    "problem": sample.problem,
                    "gold_answer": sample.gold_answer,
                    "generated_text": sample.generated_text,
                    "think_text": sample.think_text,
                    "answer_text": sample.answer_text,
                    "answer_logp_sum": sample.answer_logp_sum,
                    "answer_logp_mean": sample.answer_logp_mean,
                    "verifree_reward": sample.reward,
                    "verifree_advantage": sample.advantage,
                    "reward_source": args.reward_source,
                    "advantage_type": args.advantage_type,
                },
            )

        if step == 1 or step % args.logging_steps == 0 or step % progress_every == 0 or step == args.max_steps:
            elapsed = time.time() - start_time
            steps_per_second = step / max(elapsed, 1e-9)
            remaining = (args.max_steps - step) / max(steps_per_second, 1e-9)
            log_row = {
                "step": step,
                "max_steps": args.max_steps,
                "progress_percent": round(step / args.max_steps * 100, 2),
                "elapsed_seconds": round(elapsed, 2),
                "eta_seconds": round(remaining, 2),
                "step_seconds": round(step_time, 2),
                **{key: round(value, 6) for key, value in metrics.items()},
            }
            print(
                "[PROGRESS] "
                f"step {step}/{args.max_steps} "
                f"({log_row['progress_percent']:.2f}%) | "
                f"elapsed {format_duration(elapsed)} | "
                f"step {format_duration(step_time)} | "
                f"ETA {format_duration(remaining)} | "
                f"finish ~ {format_kst_timestamp(remaining)} | "
                f"loss {metrics['loss']:.4f} | "
                f"reward {metrics['mean_reward']:.6f}",
                flush=True,
            )
            print("[TRAIN] " + json.dumps(log_row, ensure_ascii=False, sort_keys=True), flush=True)

        if step % args.save_steps == 0 or step == args.max_steps:
            save_checkpoint(model, tokenizer, output_dir, step)
            print(f"[INFO] saved checkpoint-{step}", flush=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[DONE] saved to {output_dir}")


if __name__ == "__main__":
    main()
