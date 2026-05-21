#!/usr/bin/env python3
"""Evaluate a trained Qwen solver checkpoint on fixed Big-Math metadata.

Default behavior is deterministic greedy decoding. With --no_do_sample, changing
--batch_size should not meaningfully change accuracy. If sampling is enabled,
outputs may differ with batch size because random-number consumption changes.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from Answer_only_GRPO import (
    DEFAULT_MODEL_NAME,
    parse_completion_sections,
    render_prompt,
    verify_answer,
)


def add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    dest = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(f"--{name}", dest=dest, action="store_true", help=help_text)
    group.add_argument(f"--no_{name}", dest=dest, action="store_false", help=f"Disable: {help_text}")
    parser.set_defaults(**{dest: default})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a solver checkpoint on fixed eval metadata.")
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--adapter_path", default="")
    parser.add_argument("--eval_metadata_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_examples", type=int, default=0, help="0 means evaluate all rows.")
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    add_bool_arg(parser, "do_sample", False, "Use stochastic sampling during evaluation.")
    add_bool_arg(parser, "bf16", True, "Use bf16 when CUDA supports it.")
    add_bool_arg(parser, "load_adapter", True, "Load adapter_path as a PEFT LoRA adapter.")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_no}") from exc
    return rows


def bucket_value(row: dict[str, Any], key: str) -> str:
    return str(row.get(key, "") or "unknown")


def compute_accuracy_breakdown(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    grouped_total: Counter[str] = Counter()
    grouped_correct: Counter[str] = Counter()
    for row in rows:
        value = bucket_value(row, key)
        grouped_total[value] += 1
        grouped_correct[value] += int(bool(row.get("correct")))
    return {
        group: (grouped_correct[group] / grouped_total[group]) if grouped_total[group] else 0.0
        for group in sorted(grouped_total)
    }


def compute_group_metrics(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(bucket_value(row, key), []).append(row)

    metrics: dict[str, dict[str, Any]] = {}
    for group in sorted(grouped):
        group_rows = grouped[group]
        total = len(group_rows)
        correct_count = sum(int(bool(item.get("correct"))) for item in group_rows)
        format_success_count = sum(int(bool(item.get("exact_format_success"))) for item in group_rows)
        suspicious_count = sum(int(bool(item.get("suspicious_final_answer"))) for item in group_rows)
        metrics[group] = {
            "num_examples": total,
            "correct": correct_count,
            "accuracy": (correct_count / total) if total else 0.0,
            "format_success_rate": (format_success_count / total) if total else 0.0,
            "suspicious_final_answer_rate": (suspicious_count / total) if total else 0.0,
        }
    return metrics


def get_first_device(model: Any):
    try:
        return next(model.parameters()).device
    except StopIteration:
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch_size must be positive")
    if args.max_new_tokens <= 0:
        raise SystemExit("--max_new_tokens must be positive")
    if args.load_adapter and not str(args.adapter_path or "").strip():
        raise SystemExit("--adapter_path is required unless --no_load_adapter is set")

    random.seed(args.seed)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    eval_path = Path(args.eval_metadata_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = Path(args.adapter_path) if str(args.adapter_path or "").strip() else None

    load_adapter = bool(args.load_adapter)
    model_load_name = args.model_name
    tokenizer_load_name = args.model_name
    checkpoint_type = "base_model"
    if load_adapter:
        if adapter_path is None:
            raise SystemExit("--adapter_path is required unless --no_load_adapter is set")
        adapter_config_path = adapter_path / "adapter_config.json"
        full_model_config_path = adapter_path / "config.json"
        if adapter_config_path.exists():
            checkpoint_type = "peft_adapter"
        elif full_model_config_path.exists():
            model_load_name = str(adapter_path)
            tokenizer_load_name = str(adapter_path)
            load_adapter = False
            checkpoint_type = "full_model"
            print(
                "[INFO] adapter_path does not contain adapter_config.json; "
                "loading it as a full model checkpoint.",
                flush=True,
            )
        else:
            raise RuntimeError(
                "adapter_path is neither a PEFT adapter nor a full model checkpoint: "
                f"{adapter_path}"
            )

    rows = load_jsonl(eval_path)
    if args.max_examples and args.max_examples > 0:
        rows = rows[: args.max_examples]
    if not rows:
        raise RuntimeError(f"No eval rows loaded from {eval_path}")

    print(f"[INFO] eval rows={len(rows)} from {eval_path}")
    print(f"[INFO] model={args.model_name}")
    print(f"[INFO] loaded_model={model_load_name}")
    print(f"[INFO] adapter={args.adapter_path or '(none)'}")
    print(f"[INFO] checkpoint_type={checkpoint_type}")
    print(f"[INFO] batch_size={args.batch_size} do_sample={args.do_sample}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = None
    if torch.cuda.is_available():
        if args.bf16 and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16

    model_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(model_load_name, **model_kwargs)

    if load_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_path)
    elif args.adapter_path and Path(args.adapter_path).exists():
        print("[INFO] adapter_path is not loaded as a PEFT adapter.")

    model.eval()
    device = get_first_device(model)

    predictions_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"
    predictions_path.write_text("", encoding="utf-8")

    correct = 0
    format_success = 0
    suspicious_count = 0
    generated_lengths: list[int] = []
    result_rows: list[dict[str, Any]] = []

    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        prompts = [render_prompt(str(row.get("problem", "")), tokenizer) for row in batch_rows]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_prompt_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": args.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "do_sample": bool(args.do_sample),
        }
        if args.do_sample:
            generation_kwargs["temperature"] = max(float(args.temperature), 1e-6)
            generation_kwargs["top_p"] = float(args.top_p)

        with torch.no_grad():
            output_ids = model.generate(**inputs, **generation_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        continuation_ids = output_ids[:, prompt_len:]
        decoded = tokenizer.batch_decode(continuation_ids, skip_special_tokens=True)

        for local_idx, (row, completion) in enumerate(zip(batch_rows, decoded)):
            problem = str(row.get("problem", ""))
            gold = str(row.get("answer", ""))
            parsed = parse_completion_sections(completion)
            pred_answer = parsed.get("parsed_final_answer", "")
            verification = verify_answer(pred_answer, gold, problem_text=problem)
            is_correct = bool(verification.get("correct"))

            correct += int(is_correct)
            format_success += int(bool(parsed.get("exact_format_success")))
            suspicious_count += int(bool(parsed.get("suspicious_final_answer")))
            generated_lengths.append(len(continuation_ids[local_idx]))

            out = {
                "index": start + local_idx,
                "problem": problem,
                "gold_answer": gold,
                "source": str(row.get("source", "") or ""),
                "domain": str(row.get("domain", "") or ""),
                "difficulty_bucket": str(row.get("difficulty_bucket", "") or ""),
                "llama8b_solve_rate": row.get("llama8b_solve_rate"),
                "completion": completion,
                "parsed_final_answer": pred_answer,
                "exact_format_success": bool(parsed.get("exact_format_success")),
                "suspicious_final_answer": bool(parsed.get("suspicious_final_answer")),
                "correct": is_correct,
                "verification": verification,
            }
            result_rows.append(out)
            with predictions_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(out, ensure_ascii=False, sort_keys=True) + "\n")

        done = min(start + len(batch_rows), len(rows))
        print(f"[PROGRESS] evaluated {done}/{len(rows)}", flush=True)

    accuracy = correct / len(rows)
    source_distribution = Counter(bucket_value(row, "source") for row in result_rows)
    domain_distribution = Counter(bucket_value(row, "domain") for row in result_rows)
    difficulty_distribution = Counter(bucket_value(row, "difficulty_bucket") for row in result_rows)
    solve_rates = [
        float(row["llama8b_solve_rate"])
        for row in result_rows
        if row.get("llama8b_solve_rate") is not None
    ]
    summary = {
        "model_name": args.model_name,
        "loaded_model_name": model_load_name,
        "adapter_path": args.adapter_path,
        "checkpoint_type": checkpoint_type,
        "load_adapter": load_adapter,
        "eval_metadata_path": str(eval_path),
        "num_examples": len(rows),
        "correct": correct,
        "correct_count": correct,
        "accuracy": accuracy,
        "format_success_rate": format_success / len(rows),
        "suspicious_final_answer_rate": suspicious_count / len(rows),
        "source_distribution": dict(source_distribution),
        "domain_distribution": dict(domain_distribution),
        "difficulty_distribution": dict(difficulty_distribution),
        "accuracy_by_source": compute_accuracy_breakdown(result_rows, "source"),
        "accuracy_by_domain": compute_accuracy_breakdown(result_rows, "domain"),
        "accuracy_by_difficulty_bucket": compute_accuracy_breakdown(result_rows, "difficulty_bucket"),
        "metrics_by_source": compute_group_metrics(result_rows, "source"),
        "metrics_by_domain": compute_group_metrics(result_rows, "domain"),
        "metrics_by_difficulty_bucket": compute_group_metrics(result_rows, "difficulty_bucket"),
        "solve_rate_min": min(solve_rates) if solve_rates else None,
        "solve_rate_max": max(solve_rates) if solve_rates else None,
        "solve_rate_mean": statistics.fmean(solve_rates) if solve_rates else None,
        "batch_size": args.batch_size,
        "do_sample": bool(args.do_sample),
        "temperature": args.temperature if args.do_sample else None,
        "top_p": args.top_p if args.do_sample else None,
        "max_prompt_length": args.max_prompt_length,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "generated_length_mean": statistics.fmean(generated_lengths) if generated_lengths else None,
        "predictions_path": str(predictions_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("[DONE] eval summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
